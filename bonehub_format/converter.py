"""Converting a source dataset into a dataset of a BoneHub-format database.

A converter subclasses DatasetConverter, names the source it converts (its BoneHub ID in the registry and its
SourceInfo), and sets the handlers of `custom_data_handlers`:

    read_dataset(dataset_root) -> list[SubjectSource]   lists the subjects, their scans and their metadata
    export_image(scan, output_path)                     optional; by default the image is read from
                                                        scan.image_path, a file or a DICOM folder (see export.py)
    export_segmentation(scan, output_path)              writes the segmentation (see export.py)
    export_mesh(scan, output_folder)                    writes <LABEL>.stl files into the scan's mesh folder
    export_nurbs(scan, output_folder)                   writes <LABEL>.iges and/or <LABEL>.step files into its
                                                        nurbs folder
    export_landmarks(scan, output_folder)               writes the scan's landmarks folder: its area files, then
                                                        landmarks.json (landmarks.write_landmarks)

Where the same subject or image is also in other sources, the scan merges their data: each `Contribution` of a
ScanSource is what another source gives the scan, exported with that source's own handlers. The dataset lists every
source it draws from (other_sources), so that a source that gives all its data to the scans of others needs no
dataset of its own; its converter, if it has subjects of its own too, leaves out the subjects that live elsewhere, so
that each person is stored once.

Each data file is written with a JSON file of the same name that names its source: image.nii.gz and image.json, and
numbered files for the rest (segmentation-01.nrrd and segmentation-01.json, mesh-01/ and mesh-01.json, landmarks-01/
and landmarks-01.json, ...), the converted
source's own first, then those of the contributions in their order.

Subjects are numbered in the natural order of their source_subject_id (case_2 before case_10), and the scans of a
subject by days_from_baseline (scans without one last), then the converted source's own scans before the scans that
only other sources give, then by source_scan_id. Converting the same source again
therefore gives the same IDs. The dataset's number (ds-NNNN) is chosen by whoever builds the database.

Handlers are called with the ScanSource for the converted source's data and with the Contribution for another's;
both have source_scan_id and the *_paths fields.

Subjects are converted in worker processes, so the handlers must be module-level functions (or functools.partial
of them), and scripts that run a conversion need an `if __name__ == "__main__":` guard.
"""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import multiprocessing.connection
import os
import re
import shutil
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tqdm import tqdm

from . import layout
from ._version import SCHEMA_VERSION, is_compatible_schema_version
from .archive import model_folder_files, pack_scan
from .export import export_image as export_image_file
from .landmarks import labels as landmark_labels
from .landmarks import read_landmarks
from .metadata import DatasetInfo, FileInfo, LabeledFileInfo, Licenses, ScanInfo, SourceInfo, SubjectInfo, check_origin
from .segmentation_file import read_segmentation, read_segmentation_labels, write_indexed_segmentation

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


def export_image_from_path(scan, output_path: Path) -> None:
    """The default image handler: the image read from scan.image_path (see export.export_image)."""
    export_image_file(scan.image_path, output_path)


class CustomDataHandlers(BaseModel):
    """The functions a converter provides (see the module docstring)."""

    read_dataset: Callable[[Path], list] | None = None
    export_image: Callable[..., None] = export_image_from_path
    export_segmentation: Callable[..., None] | None = None
    export_mesh: Callable[..., None] | None = None
    export_nurbs: Callable[..., None] | None = None
    export_landmarks: Callable[..., None] | None = None

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)


def _check_origins(value):
    for origin in [value] if isinstance(value, str) else value.values():
        check_origin(origin)
    return value


class _ProvidedData(BaseModel):
    """The data one source gives a scan: where its files are, how they are named in the source, under which terms."""

    source_scan_id: str | None = Field(None, description="ID of the scan in the source")
    subset: str | None = Field(None, description="The collection of the source the scan belongs to (see SourceInfo.subsets)")
    source_split: str | None = Field(None, description="Split of the source dataset, e.g. train, val or test")
    license: Licenses | None = Field(None, description="License of this scan's data, where it differs from the source's")
    image_path: Path | None = Field(None, description="Image file or DICOM folder")
    segmentation_paths: list[Path] | None = Field(None, description="Source files of the segmentation")
    mesh_paths: list[Path] | None = Field(None, description="Source files of the meshes")
    nurbs_paths: list[Path] | None = Field(None, description="Source files of the NURBS surfaces")
    landmarks_paths: list[Path] | None = Field(None, description="Source files of the landmarks")
    segmentation_origin: Origin = Field("source", description="Origin of every label, or label -> origin")
    mesh_origin: Origin = Field("source", description="Origin of every mesh, or label -> origin")
    nurbs_origin: Origin = Field("source", description="Origin of every NURBS surface, or label -> origin")
    landmarks_origin: Origin = Field("source", description="Origin of every landmark, or label -> origin")
    remarks: str | None = Field(None, description="Remarks about this source's data of the scan, for each of its files")

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)

    @field_validator("segmentation_origin", "mesh_origin", "nurbs_origin", "landmarks_origin")
    @classmethod
    def _check_origin(cls, value):
        return _check_origins(value)

    def has_data(self) -> bool:
        return self.image_path is not None or any(
            getattr(self, f"{kind}_paths") for kind in ("segmentation", *layout.FOLDER_KINDS)
        )


class Contribution(_ProvidedData):
    """What another source gives a scan of the source being converted, exported with that source's handlers."""

    source_id: int = Field(..., ge=1, description="BoneHub ID of the source")
    source_scan_id: str = Field(..., description="ID of the scan in that source")
    handlers: CustomDataHandlers = Field(..., description="The source's export handlers")


class ScanSource(_ProvidedData):
    """One scan of a source subject: its metadata, the data the source being converted gives it, and the
    contributions of other sources. source_scan_id is None for a scan this source gives no data of its own."""

    info: ScanInfo = Field(..., description="Metadata of the scan")
    registered_to: str | None = Field(
        None, description="key of the scan of the same subject this image is registered to"
    )
    contributions: list[Contribution] = Field(default_factory=list, description="The data of other sources")

    @property
    def key(self) -> str:
        """The scan's ID among the scans read from the source: its source_scan_id, or for a scan without data of the
        converted source, '<source ID>:<source_scan_id>' of the contribution of its image (or its first)."""
        if self.source_scan_id is not None:
            return self.source_scan_id
        with_image = [c for c in self.contributions if c.image_path is not None]
        first = (with_image or self.contributions)[0]
        return f"{first.source_id}:{first.source_scan_id}"

    def check(self) -> None:
        """Raise ValueError if the scan's data does not add up."""
        if self.source_scan_id is None:
            if self.has_data():
                raise ValueError("A scan with data of the converted source needs its source_scan_id.")
            if not self.contributions:
                raise ValueError("A scan needs a source_scan_id or at least one contribution.")
        sources = [c.source_id for c in self.contributions]
        if len(set(sources)) != len(sources):
            raise ValueError(f"Scan '{self.key}' has two contributions of the same source: {sources}.")
        images = (self.image_path is not None) + sum(c.image_path is not None for c in self.contributions)
        if images > 1:
            raise ValueError(f"Scan '{self.key}' has {images} images; a scan has one.")


class SubjectSource(BaseModel):
    """One subject of the source and its scans. other_sources gives the subject's ID in the other sources whose
    contributions its scans hold."""

    info: SubjectInfo = Field(default_factory=SubjectInfo, description="gender and remarks; sources are filled in")
    source_subject_id: str = Field(..., description="ID of the subject in the converted source")
    other_sources: dict[int, str] = Field(default_factory=dict, description="BoneHub ID -> the subject's ID there")
    scans: list[ScanSource] = Field(..., min_length=1)

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)


def natural_key(text: str) -> list:
    """Sort key that orders the numbers in a text by value: case_2 before case_10."""
    return [int(part) if i % 2 else part.lower() for i, part in enumerate(re.split(r"(\d+)", text))]


def order_subjects(subjects: list[SubjectSource]) -> list[SubjectSource]:
    """Check the subjects read from a source and put them, and their scans, in the order they are numbered in."""
    if len(subjects) > layout.MAX_SUBJECTS_PER_DATASET:
        raise ValueError(
            f"read_dataset found {len(subjects)} subjects; a dataset holds at most {layout.MAX_SUBJECTS_PER_DATASET}. "
            "Split the source into several datasets."
        )
    subject_ids, scan_ids = set(), set()
    for subject in subjects:
        if not isinstance(subject, SubjectSource):
            raise TypeError(f"read_dataset must return SubjectSource objects, not {type(subject).__name__}.")
        if subject.source_subject_id in subject_ids:
            raise ValueError(f"Two subjects have the source_subject_id '{subject.source_subject_id}'.")
        subject_ids.add(subject.source_subject_id)
        if len(subject.scans) > layout.MAX_SCANS_PER_SUBJECT:
            raise ValueError(f"Subject '{subject.source_subject_id}' has more than {layout.MAX_SCANS_PER_SUBJECT} scans.")
        for scan in subject.scans:
            scan.check()
        own = {scan.key for scan in subject.scans}
        for scan in subject.scans:
            if scan.key in scan_ids:
                raise ValueError(f"Two scans have the source_scan_id '{scan.key}'.")
            scan_ids.add(scan.key)
            if scan.registered_to is not None and (scan.registered_to not in own or scan.registered_to == scan.key):
                raise ValueError(
                    f"Scan '{scan.key}' is registered to '{scan.registered_to}', which is not another scan of the "
                    "same subject."
                )
        subject.scans = sorted(
            subject.scans,
            key=lambda scan: (
                scan.info.days_from_baseline is None,
                scan.info.days_from_baseline or 0,
                scan.source_scan_id is None,  # the converted source's own scans before those of other sources
                natural_key(scan.key),
            ),
        )
    return sorted(subjects, key=lambda subject: natural_key(subject.source_subject_id))


def _origin(spec: Origin, label: str, kind: str) -> str:
    if isinstance(spec, str):
        return spec
    if label not in spec:
        raise ValueError(f"No {kind} origin given for {label}.")
    return spec[label]


def _grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return (
        np.array(reader.GetSize()),
        np.array(reader.GetSpacing()),
        np.array(reader.GetOrigin()),
        np.reshape(reader.GetDirection(), (3, 3)),
    )


def _voxel_map(image_grid: tuple, other_grid: tuple) -> np.ndarray | None:
    """Where the voxels of another grid lie on the image grid, if both hold the same voxels: the matrix A and offset
    t of image index = A @ index + t, with A a signed permutation (the same axes, maybe flipped or in another order)
    and every voxel within half a voxel of its image voxel (header rounding). None if the grids differ otherwise."""
    size, spacing, origin, direction = image_grid
    other_size, other_spacing, other_origin, other_direction = other_grid
    to_index = np.linalg.inv(direction * spacing)  # direction * spacing scales the columns: index -> physical
    matrix = to_index @ (other_direction * other_spacing)
    axes = np.round(matrix)
    if np.abs(matrix - axes).max() > 1e-3 or not np.array_equal(np.abs(axes) @ np.abs(axes).T, np.eye(3)):
        return None
    offset = to_index @ (other_origin - origin)
    if np.abs(offset - np.round(offset)).max() >= 0.5:
        return None
    # every voxel of the other grid lands on a voxel of the image, and the other way round
    corner = axes @ (other_size - 1)
    lands = np.round(offset) + np.minimum(corner, 0), np.round(offset) + np.maximum(corner, 0)
    if not (np.array_equal(lands[0], np.zeros(3)) and np.array_equal(lands[1], size - 1)):
        return None
    return np.column_stack([matrix, offset])


def _align_to_image(scan_path: Path) -> None:
    """Put each segmentation on the grid of the scan's image, so that the data of all sources describe the same
    voxels. A segmentation that holds the image's voxels in another order (a flipped axis) or with a slightly
    different origin (less than half a voxel: header rounding) is written again on the image grid; nearest-neighbour
    resampling then moves each voxel to its image voxel without changing any label. Raise if a segmentation is on
    another grid."""
    image = scan_path / layout.IMAGE_FILE
    if not image.is_file():
        return
    image_grid = _grid(image)
    for entry in sorted(scan_path.iterdir()):
        parsed = layout.parse_scan_entry(entry.name)
        if parsed is None or parsed[0] != "segmentation" or parsed[2] != "data":
            continue
        path = entry
        other_grid = _grid(path)
        voxel_map = _voxel_map(image_grid, other_grid)
        if voxel_map is None:
            raise ValueError(
                f"{path.name} in '{scan_path}' is not on the grid of the scan's image: size, spacing, "
                f"origin and direction are {tuple(v.ravel().tolist() for v in other_grid)} instead of "
                f"{tuple(v.ravel().tolist() for v in image_grid)}."
            )
        if np.abs(voxel_map - np.column_stack([np.eye(3), np.zeros(3)])).max() < 1e-3:
            continue  # the same grid, to within 1/1000 of a voxel
        size, spacing, origin, direction = image_grid
        layers, segments = read_segmentation(path)
        aligned = [
            sitk.Resample(
                numbers,
                [int(v) for v in size],
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                origin.tolist(),
                spacing.tolist(),
                direction.ravel().tolist(),
                0,
                numbers.GetPixelID(),
            )
            for numbers in layers
        ]
        write_indexed_segmentation([sitk.GetArrayViewFromImage(a) for a in aligned], segments, aligned[0], path)


def _exit_when_parent_dies() -> None:
    """Hard-exit this worker once the parent process is gone.

    Stopping a conversion (e.g. a test run in an IDE) can kill the parent without shutting the pool down; the orphaned
    workers would otherwise finish their queued subjects, write into a folder the next run is converting, and then
    hang on the log queue nobody reads.
    """
    parent = multiprocessing.parent_process()
    if parent is None:
        return
    multiprocessing.connection.wait([parent.sentinel])
    os._exit(1)


def _init_subject_worker(log_queue, logger_name: str) -> None:
    """Forward the worker's log records and warnings to the parent, keep ITK single-threaded, and exit with the parent."""
    threading.Thread(target=_exit_when_parent_dies, name="parent-watchdog", daemon=True).start()
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
    Base class for converting a source dataset into a dataset of a BoneHub-format database.
    Args:
        dataset_root (Path): The root directory of the source dataset.
        source_id (int): The source's BoneHub ID in the registry.
        source (SourceInfo): The source and how it is converted.
        other_sources (dict): BoneHub ID -> SourceInfo of every other source whose data the scans take in
            (Contribution.source_id).
        name (str): Name of the dataset; the source's by default.
        description (str): How the dataset is made from its sources, where it is not one source.
    The dataset's information (dataset.json) is `dataset_info`.
    """

    # Default number of subjects converted at once; converters that need much memory per subject set fewer.
    max_workers: int = DEFAULT_MAX_WORKERS

    def __init__(
        self,
        dataset_root: Path,
        source_id: int,
        source: SourceInfo,
        other_sources: dict[int, SourceInfo] | None = None,
        name: str | None = None,
        description: str | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.source_id = source_id
        self.dataset_info = DatasetInfo(
            name=name or source.name, description=description, sources={source_id: source, **(other_sources or {})}
        )
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
            output_dataset_id (int): The dataset's number in the database; the folder is ds-<number, 4 digits>.
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
        if not subjects:
            raise ValueError("The source lists no subjects; a dataset without subjects is not written.")
        used = {c.source_id for subject in subjects for scan in subject.scans for c in scan.contributions}
        used |= {source_id for subject in subjects for source_id in subject.other_sources}
        missing = sorted(used - set(self.dataset_info.sources))
        if missing:
            raise ValueError(f"The scans take in data of sources {missing}, which other_sources does not describe.")

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
            source = subject.source_subject_id
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
            previous = SubjectInfo.load(subject_json).sources.get(self.source_id)
            expected = subjects[subject_id - 1].source_subject_id
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
        scan_names = {scan.key: layout.scan_dir_name(i) for i, scan in enumerate(subject.scans, start=1)}
        try:
            for scan in subject.scans:
                self._convert_scan(scan, subject_path / scan_names[scan.key], scan_names)
            info = subject.info.model_copy(deep=True)
            info.sources = {**subject.other_sources, self.source_id: subject.source_subject_id}
            info.save(subject_path / layout.SUBJECT_JSON)
        except BaseException:
            shutil.rmtree(subject_path, ignore_errors=True)
            raise
        self.logger.info(f"Converted subject {subject_id} ('{subject.source_subject_id}', {len(subject.scans)} scans).")

    def _convert_scan(self, scan: ScanSource, scan_path: Path, scan_names: dict[str, str]) -> None:
        scan_path.mkdir(parents=True)
        info = scan.info  # the handlers may add to it, e.g. remarks on how the image was made
        parts = [(self.source_id, scan, self.custom_data_handlers)] if scan.source_scan_id is not None else []
        parts += [(c.source_id, c, c.handlers) for c in scan.contributions]
        numbers = dict.fromkeys(layout.LABELED_KINDS, 0)  # the files of each kind written so far
        for source_id, data, handlers in parts:
            self._export_provided(data, handlers, scan_path, source_id, numbers)
        _align_to_image(scan_path)
        if scan.registered_to is not None:
            info.registered_to = scan_names[scan.registered_to]
        pack_scan(scan_path)
        info.save(scan_path / layout.SCAN_JSON)

    def _export_provided(
        self, data, handlers: CustomDataHandlers, scan_path: Path, source_id: int, numbers: dict[str, int]
    ) -> None:
        """Write the files one source gives a scan, each with its JSON file; number them after those of the sources
        before (`numbers`, updated)."""
        where = f"Scan '{data.source_scan_id}' of source {source_id}"

        def file_info(kind: str, model=FileInfo):
            return model(
                source=source_id,
                source_scan_id=data.source_scan_id,
                subset=data.subset,
                source_split=data.source_split,
                license=data.license.of(kind) if data.license else None,
                remarks=data.remarks,
            )

        if data.image_path is not None:
            image_path = layout.data_file(scan_path, "image")
            if image_path.exists():
                raise ValueError(f"{where} gives an image, but the scan has one already.")
            handlers.export_image(data, image_path)
            if not image_path.is_file():
                raise RuntimeError(f"export_image wrote no image for {where}.")
            file_info("image").save(layout.data_json(scan_path, "image"))

        if data.segmentation_paths:
            if handlers.export_segmentation is None:
                raise ValueError(f"{where} has segmentation files, but no export_segmentation handler is set.")
            number = numbers["segmentation"] + 1
            segmentation_path = layout.data_file(scan_path, "segmentation", number)
            handlers.export_segmentation(data, segmentation_path)
            labels = read_segmentation_labels(segmentation_path) if segmentation_path.is_file() else []
            if labels:
                info = file_info("segmentation", LabeledFileInfo)
                info.set_labels(labels, {label: _origin(data.segmentation_origin, label, "segmentation") for label in labels})
                info.save(layout.data_json(scan_path, "segmentation", number))
                numbers["segmentation"] = number
            else:
                segmentation_path.unlink(missing_ok=True)
                self.logger.warning(f"{where}: its segmentation holds no labels and is left out.")

        for kind in layout.FOLDER_KINDS:
            if not getattr(data, f"{kind}_paths"):
                continue
            handler = getattr(handlers, f"export_{kind}")
            if handler is None:
                raise ValueError(f"{where} has {kind} files, but no export_{kind} handler is set.")
            number = numbers[kind] + 1
            folder = layout.model_folder(scan_path, kind, number)
            folder.mkdir()
            handler(data, folder)
            files = model_folder_files(folder, kind)
            if kind == "landmarks":
                labels = landmark_labels(read_landmarks(folder)) if files else []
            else:
                labels = sorted({path.stem for path in files})
            if labels:
                origin = getattr(data, f"{kind}_origin")
                info = file_info(kind, LabeledFileInfo)
                info.set_labels(labels, {label: _origin(origin, label, kind) for label in labels})
                info.save(layout.data_json(scan_path, kind, number))
                numbers[kind] = number
            else:
                shutil.rmtree(folder)
                self.logger.warning(f"{where}: no {kind} files were written.")
