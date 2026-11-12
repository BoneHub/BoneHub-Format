"""Converting a source dataset into a BoneHub-format dataset.

A converter subclasses DatasetConverter and sets the handlers of `custom_data_handlers`:

    read_dataset(dataset_root) -> list[SubjectSource]   lists the subjects, their scans and their metadata
    export_image(scan, output_path)                     optional; by default the image is read from
                                                        scan.image_path, a file or a DICOM folder (see export.py)
    export_segmentation(scan, output_path)              writes segmentation.seg.nrrd (see export.py)
    export_mesh(scan, output_folder)                    writes <LABEL>.stl files into the scan's mesh/ folder
    export_nurbs(scan, output_folder)                   writes <LABEL>.iges and/or <LABEL>.step files into nurbs/

Subjects are numbered in the natural order of their source_subject_id (case_2 before case_10), and the scans of a
subject by days_from_baseline (scans without one last) and then by source_scan_id. Converting the same source again
therefore gives the same IDs, which is what lets derived datasets and publications refer to them. The dataset ID is
the dataset's BoneHub ID in the registry (see registry.py).

Subjects are converted in worker processes, so the handlers must be module-level functions (or functools.partial
of them), and scripts that run a conversion need an `if __name__ == "__main__":` guard.
"""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import os
import re
import shutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from pathlib import Path

import SimpleITK as sitk
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tqdm import tqdm

from . import layout
from ._version import SCHEMA_VERSION, is_compatible_schema_version
from .archive import model_folder_files, pack_scan
from .export import export_image as export_image_file
from .metadata import DatasetInfo, ScanInfo, SubjectInfo, check_origin
from .segmentation_file import read_segmentation_labels

# Each worker process holds whole volumes in memory, so keep the default modest.
DEFAULT_MAX_WORKERS = 8


class DatasetConversionError(RuntimeError):
    """Raised by export_to_bonehub_format when subjects failed or the conversion stopped; details are in the log."""

    def __init__(self, message: str, log_file_path: Path, failed_subject_ids: tuple[int, ...] = ()):
        # All fields go to args so the error survives pickling back from a worker process.
        super().__init__(message, log_file_path, failed_subject_ids)
        self.message = message
        self.log_file_path = log_file_path
        self.failed_subject_ids = failed_subject_ids

    def __str__(self) -> str:
        return f"{self.message} See '{self.log_file_path}' for details."


Origin = str | dict[str, str]


class ScanSource(BaseModel):
    """One scan of a source subject: where its data is, and its metadata."""

    info: ScanInfo = Field(..., description="Metadata of the scan; the label fields are filled in by the converter")
    image_path: Path | None = Field(None, description="Image file or DICOM folder")
    segmentation_paths: list[Path] | None = Field(None, description="Source files of the segmentation")
    mesh_paths: list[Path] | None = Field(None, description="Source files of the meshes")
    nurbs_paths: list[Path] | None = Field(None, description="Source files of the NURBS surfaces")
    registered_to: str | None = Field(
        None, description="source_scan_id of the scan of the same subject this image is registered to"
    )
    segmentation_origin: Origin = Field("source", description="Origin of every label, or label -> origin")
    mesh_origin: Origin = Field("source", description="Origin of every mesh, or label -> origin")
    nurbs_origin: Origin = Field("source", description="Origin of every NURBS surface, or label -> origin")

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)

    @field_validator("segmentation_origin", "mesh_origin", "nurbs_origin")
    @classmethod
    def _check_origin(cls, value):
        for origin in [value] if isinstance(value, str) else value.values():
            check_origin(origin)
        return value


class SubjectSource(BaseModel):
    """One subject of the source dataset and its scans."""

    info: SubjectInfo
    scans: list[ScanSource] = Field(..., min_length=1)

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)


def export_image_from_path(scan: ScanSource, output_path: Path) -> None:
    """The default image handler: the image read from scan.image_path (see export.export_image)."""
    export_image_file(scan.image_path, output_path)


class CustomDataHandlers(BaseModel):
    """The functions a converter provides (see the module docstring)."""

    read_dataset: Callable[[Path], list[SubjectSource]] | None = None
    export_image: Callable[[ScanSource, Path], None] = export_image_from_path
    export_segmentation: Callable[[ScanSource, Path], None] | None = None
    export_mesh: Callable[[ScanSource, Path], None] | None = None
    export_nurbs: Callable[[ScanSource, Path], None] | None = None

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)


def natural_key(text: str) -> list:
    """Sort key that orders the numbers in a text by value: case_2 before case_10."""
    return [int(part) if i % 2 else part.lower() for i, part in enumerate(re.split(r"(\d+)", text))]


def order_subjects(subjects: list[SubjectSource]) -> list[SubjectSource]:
    """Check the subjects read from a source and put them, and their scans, in the order they are numbered in."""
    if not subjects:
        raise ValueError("read_dataset found no subjects.")
    if len(subjects) > layout.MAX_SUBJECTS_PER_DATASET:
        raise ValueError(
            f"read_dataset found {len(subjects)} subjects; a dataset holds at most {layout.MAX_SUBJECTS_PER_DATASET}. "
            "Split the source into several datasets."
        )
    subject_ids, scan_ids = set(), set()
    for subject in subjects:
        if not isinstance(subject, SubjectSource):
            raise TypeError(f"read_dataset must return SubjectSource objects, not {type(subject).__name__}.")
        if subject.info.source_subject_id in subject_ids:
            raise ValueError(f"Two subjects have the source_subject_id '{subject.info.source_subject_id}'.")
        subject_ids.add(subject.info.source_subject_id)
        if len(subject.scans) > layout.MAX_SCANS_PER_SUBJECT:
            raise ValueError(f"Subject '{subject.info.source_subject_id}' has more than {layout.MAX_SCANS_PER_SUBJECT} scans.")
        own = {scan.info.source_scan_id for scan in subject.scans}
        for scan in subject.scans:
            if scan.info.source_scan_id in scan_ids:
                raise ValueError(f"Two scans have the source_scan_id '{scan.info.source_scan_id}'.")
            scan_ids.add(scan.info.source_scan_id)
            if scan.registered_to is not None and (
                scan.registered_to not in own or scan.registered_to == scan.info.source_scan_id
            ):
                raise ValueError(
                    f"Scan '{scan.info.source_scan_id}' is registered to '{scan.registered_to}', which is not another "
                    "scan of the same subject."
                )
        subject.scans = sorted(
            subject.scans,
            key=lambda scan: (
                scan.info.days_from_baseline is None,
                scan.info.days_from_baseline or 0,
                natural_key(scan.info.source_scan_id),
            ),
        )
    return sorted(subjects, key=lambda subject: natural_key(subject.info.source_subject_id))


def _origin(spec: Origin, label: str, kind: str) -> str:
    if isinstance(spec, str):
        return spec
    if label not in spec:
        raise ValueError(f"No {kind} origin given for {label}.")
    return spec[label]


def _init_subject_worker(log_queue, logger_name: str) -> None:
    """Forward the worker's log records and warnings to the parent and keep ITK single-threaded."""
    for name in (logger_name, "py.warnings"):
        logger = logging.getLogger(name)
        logger.handlers = [logging.handlers.QueueHandler(log_queue)]
        logger.setLevel(logging.INFO)
        logger.propagate = False
    logging.captureWarnings(True)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


def _remove_dataset_folder(dataset_path: Path) -> None:
    """Delete a dataset folder to convert it again, if it is one: it holds a dataset.json, or nothing."""
    if any(dataset_path.iterdir()) and not (dataset_path / layout.DATASET_JSON).is_file():
        raise FileExistsError(f"'{dataset_path}' exists and is not a BoneHub dataset folder; it was left as it is.")
    shutil.rmtree(dataset_path)


def _format_ids(ids: list[int], limit: int = 20) -> str:
    """List the ids, shortened when there are many; the log always has the full list."""
    if len(ids) <= limit:
        return str(ids)
    return f"{str(ids[:limit])[:-1]}, ... ({len(ids) - limit} more)]"


class DatasetConverter:
    """
    Base class for converting a source dataset into a BoneHub-format dataset.
    Args:
        dataset_root (Path): The root directory of the source dataset.
        dataset_info (DatasetInfo): Information about the dataset, written to dataset.json.
    """

    # Default number of subjects converted at once; converters that need much memory per subject set fewer.
    max_workers: int = DEFAULT_MAX_WORKERS

    def __init__(self, dataset_root: Path, dataset_info: DatasetInfo):
        self.dataset_root = Path(dataset_root)
        self.dataset_info = dataset_info
        self.custom_data_handlers = CustomDataHandlers()
        self.logger = logging.getLogger(self.__class__.__name__)

    def read_subjects(self) -> list[SubjectSource]:
        """The subjects of the source in the order they are numbered: subject k is sub-k."""
        if self.custom_data_handlers.read_dataset is None:
            raise NotImplementedError(f"{self.__class__.__name__} sets no read_dataset handler.")
        return order_subjects(self.custom_data_handlers.read_dataset(self.dataset_root))

    def export_to_bonehub_format(
        self,
        output_root: Path,
        output_dataset_id: int,
        overwrite: bool = False,
        resume: bool = False,
        num_workers: int | None = None,
        verbose: bool = True,
    ) -> Path:
        """
        Convert the source into <output_root>/ds-<output_dataset_id>.
        Args:
            output_root (Path): The root folder of the database to write the dataset into.
            output_dataset_id (int): The dataset's BoneHub ID in the registry; the folder is ds-<ID, 4 digits>.
            overwrite (bool): Delete an existing dataset folder of this ID and convert everything again.
            resume (bool): Keep the subjects an earlier run of this conversion finished and convert the rest, e.g.
                those that failed. The source must still list the same subjects.
            num_workers (int): Subjects converted at once, each in its own process; 1 converts them one after the
                other in this process. Default: the number of CPU cores, at most the class's max_workers.
            verbose (bool): Show a progress bar.
        Returns:
            Path: The log file of this conversion, in <output_root>/logs.
        Raises:
            DatasetConversionError: If any subject failed (the others are still converted) or the conversion stopped
                early. The log file names the failed subjects with their tracebacks.
        """
        if overwrite and resume:
            raise ValueError("Choose overwrite (convert everything again) or resume (finish an earlier run), not both.")
        output_root = Path(output_root)
        dataset_path = layout.dataset_path(output_root, output_dataset_id)
        if dataset_path.exists():
            if overwrite:
                _remove_dataset_folder(dataset_path)
            elif not resume:
                raise FileExistsError(
                    f"'{dataset_path}' already exists. Pass overwrite=True to convert it again, or resume=True to "
                    "finish an earlier run."
                )

        log_dir = output_root / layout.LOGS_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / f"{dataset_path.name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        log_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
        log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(log_handler)
        self.logger.setLevel(logging.INFO)
        try:
            failed_subject_ids = self._export(dataset_path, log_handler, num_workers, resume, verbose)
        except Exception as exc:
            self.logger.exception(f"Conversion of {dataset_path.name} stopped by an error.")
            raise DatasetConversionError(f"Conversion of {dataset_path.name} stopped: {exc!r}.", log_file_path) from exc
        finally:
            self.logger.removeHandler(log_handler)
            log_handler.close()

        if failed_subject_ids:
            raise DatasetConversionError(
                f"{len(failed_subject_ids)} subject(s) of {dataset_path.name} failed: {_format_ids(failed_subject_ids)}.",
                log_file_path,
                tuple(failed_subject_ids),
            )
        return log_file_path

    def _export(
        self, dataset_path: Path, log_handler: logging.Handler, num_workers: int | None, resume: bool, verbose: bool
    ) -> list[int]:
        """Convert every subject into dataset_path and return the IDs of the subjects that failed."""
        self.logger.info(f"Reading the source dataset in '{self.dataset_root}'...")
        subjects = self.read_subjects()
        scan_count = sum(len(subject.scans) for subject in subjects)
        self.logger.info(f"Found {len(subjects)} subjects with {scan_count} scans.")

        finished = self._finished_subjects(dataset_path, subjects) if resume and dataset_path.exists() else set()
        if finished:
            self.logger.info(f"Keeping {len(finished)} subjects converted by an earlier run.")

        info = self.dataset_info.model_copy(deep=True)
        info.schema_version = SCHEMA_VERSION
        info.save(dataset_path / layout.DATASET_JSON)
        self.logger.info(f"Writing '{dataset_path}'...")

        pending = [(subject_id, subject) for subject_id, subject in enumerate(subjects, start=1) if subject_id not in finished]
        if num_workers is None:
            num_workers = min(os.cpu_count() or 4, self.max_workers)
        num_workers = max(1, min(num_workers, len(pending)))

        failed = []
        progress = tqdm(total=len(pending), desc=f"Converting {dataset_path.name}", unit="subject", disable=not verbose)

        def record_failure(subject_id: int, subject: SubjectSource, exc: BaseException) -> None:
            failed.append(subject_id)
            progress.set_postfix(failed=len(failed))
            source = subject.info.source_subject_id
            if isinstance(exc, BrokenProcessPool):
                # A worker died without raising, and every unfinished subject ends here; no traceback helps.
                self.logger.error(
                    f"Subject {subject_id} failed (source: '{source}'): a worker process died unexpectedly, often "
                    "from running out of memory."
                )
            else:
                self.logger.error(f"Subject {subject_id} failed (source: '{source}').", exc_info=exc)

        if num_workers == 1:
            for subject_id, subject in pending:
                try:
                    self._convert_subject(subject_id, subject, dataset_path)
                except Exception as exc:
                    record_failure(subject_id, subject, exc)
                progress.update()
        else:
            # Worker processes are spawned, so scripts calling this need an `if __name__ == "__main__":` guard.
            context = multiprocessing.get_context("spawn")
            log_queue = context.Queue()
            log_listener = logging.handlers.QueueListener(log_queue, log_handler)
            log_listener.start()
            try:
                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    mp_context=context,
                    initializer=_init_subject_worker,
                    initargs=(log_queue, self.logger.name),
                ) as executor:
                    futures = {
                        executor.submit(self._convert_subject, subject_id, subject, dataset_path): (subject_id, subject)
                        for subject_id, subject in pending
                    }
                    for future in as_completed(futures):
                        subject_id, subject = futures[future]
                        try:
                            future.result()
                        except Exception as exc:
                            record_failure(subject_id, subject, exc)
                        progress.update()
            finally:
                log_listener.stop()
        progress.close()

        failed.sort()
        self.logger.info(
            f"Summary of {dataset_path.name}: {len(subjects)} subjects; {len(pending) - len(failed)} converted, "
            f"{len(finished)} kept from an earlier run, {len(failed)} failed."
        )
        if failed:
            self.logger.error(
                f"Failed subjects: {failed}. Their errors are logged above; run again with resume=True to convert "
                "only the subjects that are missing."
            )
        return failed

    def _finished_subjects(self, dataset_path: Path, subjects: list[SubjectSource]) -> set[int]:
        """The subjects an earlier run finished, after checking that they are the subjects the source lists now."""
        info_path = dataset_path / layout.DATASET_JSON
        if not info_path.is_file() or not is_compatible_schema_version(DatasetInfo.load(info_path).schema_version):
            raise ValueError(f"'{dataset_path}' is not from an earlier run of this conversion; use overwrite=True.")
        finished = set()
        for entry in dataset_path.iterdir():
            if entry.name == layout.DATASET_JSON:
                continue
            subject_id = layout.parse_subject_dir(entry.name)
            if subject_id is None:
                raise ValueError(f"'{entry}' does not belong in a dataset folder; use overwrite=True.")
            if subject_id > len(subjects):
                raise ValueError(
                    f"'{dataset_path}' has subject {subject_id}, but the source lists only {len(subjects)} subjects; "
                    "use overwrite=True."
                )
            subject_json = entry / layout.SUBJECT_JSON
            if not subject_json.is_file():
                continue  # not finished: it is converted again
            previous = SubjectInfo.load(subject_json).source_subject_id
            expected = subjects[subject_id - 1].info.source_subject_id
            if previous != expected:
                raise ValueError(
                    f"'{entry.name}' was converted from '{previous}', but the source now lists '{expected}' as subject "
                    f"{subject_id}; use overwrite=True."
                )
            finished.add(subject_id)
        return finished

    def _convert_subject(self, subject_id: int, subject: SubjectSource, dataset_path: Path) -> None:
        """Write a subject folder; subject.json is written last, so a folder without it is unfinished."""
        subject_path = dataset_path / layout.subject_dir_name(subject_id)
        if subject_path.exists():
            shutil.rmtree(subject_path)  # left unfinished by an earlier run
        scan_names = {scan.info.source_scan_id: layout.scan_dir_name(i) for i, scan in enumerate(subject.scans, start=1)}
        try:
            for scan in subject.scans:
                self._convert_scan(scan, subject_path / scan_names[scan.info.source_scan_id], scan_names)
            subject.info.save(subject_path / layout.SUBJECT_JSON)
        except BaseException:
            shutil.rmtree(subject_path, ignore_errors=True)
            raise
        self.logger.info(f"Converted subject {subject_id} ('{subject.info.source_subject_id}', {len(subject.scans)} scans).")

    def _convert_scan(self, scan: ScanSource, scan_path: Path, scan_names: dict[str, str]) -> None:
        scan_path.mkdir(parents=True)
        handlers = self.custom_data_handlers
        info = scan.info  # the handlers may add to it, e.g. remarks on how the image was made

        if scan.image_path is not None:
            image_path = scan_path / layout.IMAGE_FILE
            handlers.export_image(scan, image_path)
            if not image_path.is_file():
                raise RuntimeError(f"export_image wrote no image for '{info.source_scan_id}'.")

        if scan.segmentation_paths:
            if handlers.export_segmentation is None:
                raise ValueError(f"Scan '{info.source_scan_id}' has segmentation files, but no export_segmentation handler is set.")
            segmentation_path = scan_path / layout.SEGMENTATION_FILE
            handlers.export_segmentation(scan, segmentation_path)
            labels = read_segmentation_labels(segmentation_path) if segmentation_path.is_file() else []
            if not labels:
                segmentation_path.unlink(missing_ok=True)
                self.logger.warning(f"Scan '{info.source_scan_id}': its segmentation holds no labels and is left out.")
            info.set_labels("segmentation", labels, {label: _origin(scan.segmentation_origin, label, "segmentation") for label in labels})

        for kind in layout.MODEL_KINDS:
            if not getattr(scan, f"{kind}_paths"):
                continue
            handler = getattr(handlers, f"export_{kind}")
            if handler is None:
                raise ValueError(f"Scan '{info.source_scan_id}' has {kind} files, but no export_{kind} handler is set.")
            folder = layout.model_folder(scan_path, kind)
            folder.mkdir()
            handler(scan, folder)
            labels = sorted({path.stem for path in model_folder_files(folder, kind)})
            if not labels:
                shutil.rmtree(folder)
                self.logger.warning(f"Scan '{info.source_scan_id}': no {kind} files were written.")
            origin = getattr(scan, f"{kind}_origin")
            info.set_labels(kind, labels, {label: _origin(origin, label, kind) for label in labels})

        if scan.registered_to is not None:
            info.registered_to = scan_names[scan.registered_to]
        pack_scan(scan_path)
        info.save(scan_path / layout.SCAN_JSON)
