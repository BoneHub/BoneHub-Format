"""Checking that a database follows the BoneHub format.

    problems = validate(root)                         # the data itself
    problems = validate(root, registry=REGISTRY_URL)  # and that every source is a dataset of the registry
    problems = validate(root, publishing=True)        # and that it is ready to upload: licenses, zips and index

Each problem is a line of text naming the file or folder concerned; an empty list means none was found.
"""

import zipfile
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import ValidationError

from . import layout
from ._version import SCHEMA_VERSION, is_compatible_schema_version
from .archive import model_folder_files, zip_is_current
from .metadata import DatasetInfo, FileInfo, LabeledFileInfo, ScanInfo, SubjectInfo, file_license
from .registry import as_registry
from .segmentation_file import read_segmentation_labels

_INDEX_FILES = ("datasets.parquet", "sources.parquet", "scans.parquet", "files.parquet", "labels.parquet")
#: Subjects checked at once: the time is spent waiting for the disk, often a network share, not computing.
READ_THREADS = 16


def _load(model, path: Path, problems: list[str]):
    try:
        return model.load(path)
    except (ValidationError, ValueError, OSError) as e:
        problems.append(f"{path}: {e}")
        return None


def _model_labels(
    scan_folder: Path, kind: str, number: int, problems: list[str], publishing: bool, zips_checked: bool = False
) -> set[str] | None:
    """The labels a model file of a scan holds; None if they cannot be read. The zip is compared with its folder
    unless zips_checked (pack has just done so)."""
    folder = layout.model_folder(scan_folder, kind, number)
    zip_path = layout.model_zip(scan_folder, kind, number)
    try:
        if folder.is_dir():
            labels = {f.stem for f in model_folder_files(folder, kind)}
            if labels and zip_path.is_file() and not zips_checked and not zip_is_current(folder, zip_path, kind):
                problems.append(f"{zip_path}: does not match {folder.name}/; run pack.")
            elif labels and publishing and not zip_path.is_file():
                problems.append(f"{folder}: has no {zip_path.name}, which is what gets published; run pack.")
            return labels
        if zip_path.is_file():
            with zipfile.ZipFile(zip_path) as archive:
                return {Path(name).stem for name in archive.namelist()}
    except (ValueError, OSError) as e:
        problems.append(f"{folder}: {e}")
        return None
    return set()


class _Checker:
    def __init__(self, registry: Mapping[int, str] | None, publishing: bool, zips_checked: bool = False):
        self.registry = registry
        self.publishing = publishing
        self.zips_checked = zips_checked
        self.problems: list[str] = []
        self.missing_licenses: Counter = Counter()  # (dataset folder, source, kind) -> files without a known license

    def load_dataset(self, dataset_folder: Path) -> DatasetInfo | None:
        """Read and check a dataset.json."""
        dataset_json = dataset_folder / layout.DATASET_JSON
        info = _load(DatasetInfo, dataset_json, self.problems)
        if info is None:
            return None
        if not is_compatible_schema_version(info.schema_version):
            self.problems.append(
                f"{dataset_json}: written in format version {info.schema_version}, which this package (format version "
                f"{SCHEMA_VERSION}) cannot read."
            )
            return None
        for source_id in info.sources:
            if self.registry is not None and source_id not in self.registry:
                self.problems.append(f"{dataset_json}: source {source_id} is not a BoneHub ID of the registry.")
        return info

    def check_dataset(self, dataset_folder: Path, dataset: DatasetInfo) -> None:
        problems = self.problems
        subject_folders = []
        for entry in sorted(dataset_folder.iterdir()):
            if entry.name == layout.DATASET_JSON:
                continue
            if layout.parse_subject_dir(entry.name) is None or not entry.is_dir():
                problems.append(f"{entry}: not a subject folder (sub-NNNNNN).")
                continue
            subject_folders.append(entry)
        if not subject_folders:
            problems.append(f"{dataset_folder}: has no subjects.")

        def check(subject_folder: Path) -> "_Checker":
            checker = _Checker(self.registry, self.publishing, self.zips_checked)
            checker.check_subject(subject_folder, dataset)
            return checker

        # each subject on its own, read in parallel; the problems are kept in the order of the subjects
        with ThreadPoolExecutor(max_workers=READ_THREADS) as pool:
            for checker in pool.map(check, subject_folders):
                problems += checker.problems
                self.missing_licenses.update(checker.missing_licenses)

    def check_subject(self, subject_folder: Path, dataset: DatasetInfo) -> None:
        problems = self.problems
        subject_json = subject_folder / layout.SUBJECT_JSON
        subject = _load(SubjectInfo, subject_json, problems)
        if subject is None:
            return
        if not subject.sources:
            problems.append(f"{subject_json}: gives the subject's ID in no source.")
        for source_id in subject.sources:
            if source_id not in dataset.sources:
                problems.append(f"{subject_json}: names source {source_id}, which dataset.json does not list.")
        scan_folders = []
        for entry in sorted(subject_folder.iterdir()):
            if entry.name == layout.SUBJECT_JSON:
                continue
            if layout.parse_scan_dir(entry.name) is None or not entry.is_dir():
                problems.append(f"{entry}: not a scan folder (scan-NN).")
                continue
            scan_folders.append(entry)
        if not scan_folders:
            problems.append(f"{subject_folder}: has no scans.")
        names = {folder.name for folder in scan_folders}
        for scan_folder in scan_folders:
            if not (scan_folder / layout.SCAN_JSON).is_file():
                problems.append(f"{scan_folder}: has no {layout.SCAN_JSON}.")
                continue
            self.check_scan(scan_folder, names, dataset)

    def check_scan(self, scan_folder: Path, sibling_names: set[str], dataset: DatasetInfo) -> None:
        problems = self.problems
        scan_json = scan_folder / layout.SCAN_JSON
        info = _load(ScanInfo, scan_json, problems)
        if info is None:
            return
        if info.registered_to is not None and (info.registered_to not in sibling_names or info.registered_to == scan_folder.name):
            problems.append(f"{scan_json}: registered_to names '{info.registered_to}', not another scan of this subject.")

        roles: dict[tuple[str, int | None], set[str]] = {}
        for entry in scan_folder.iterdir():
            if entry.name == layout.SCAN_JSON:
                continue
            parsed = layout.parse_scan_entry(entry.name)
            if parsed is None or (parsed[2] == "folder") != entry.is_dir():
                problems.append(f"{entry}: not part of a scan folder.")
                continue
            kind, number, role = parsed
            roles.setdefault((kind, number), set()).add(role)
        if not roles:
            problems.append(f"{scan_folder}: holds no data.")

        for kind in layout.LABELED_KINDS:
            numbers = sorted(number for k, number in roles if k == kind)
            if numbers != list(range(1, len(numbers) + 1)):
                problems.append(f"{scan_folder}: its {kind} files are numbered {numbers}, not 1 to {len(numbers)}.")

        for (kind, number), found in sorted(roles.items(), key=lambda item: (layout.DATA_KINDS.index(item[0][0]), item[0][1] or 0)):
            json_path = layout.data_json(scan_folder, kind, number)
            data_path = layout.data_file(scan_folder, kind, number)
            if "json" not in found:
                problems.append(f"{scan_folder}: has {data_path.name} without {json_path.name}.")
                continue
            if not found & {"data", "folder"}:
                problems.append(f"{json_path}: describes {data_path.name}, which is missing.")
                continue
            file_info = _load(FileInfo if kind == "image" else LabeledFileInfo, json_path, problems)
            if file_info is None:
                continue
            source = dataset.source(file_info.source)
            if source is None:
                problems.append(f"{json_path}: comes from source {file_info.source}, which dataset.json does not list.")
                continue
            if file_info.subset is not None and file_info.subset not in (source.subsets or {}):
                problems.append(f"{json_path}: names subset '{file_info.subset}', which source {file_info.source} lacks.")
            if kind == "segmentation":
                self._check_segmentation(data_path, set(file_info.labels))
            elif kind in layout.MODEL_KINDS:
                labels = _model_labels(scan_folder, kind, number, problems, self.publishing, self.zips_checked)
                if labels is not None and labels != set(file_info.labels):
                    problems.append(f"{json_path}: lists {sorted(file_info.labels)}, but the files are of {sorted(labels)}.")
            if self.publishing and file_license(kind, file_info, dataset) is None:
                self.missing_licenses[(scan_folder.parent.parent.name, file_info.source, kind)] += 1

    def _check_segmentation(self, path: Path, expected: set[str]) -> None:
        try:
            found = set(read_segmentation_labels(path))
        except (ValueError, OSError) as e:
            self.problems.append(f"{path}: {e}")
            return
        if not found:
            self.problems.append(f"{path}: holds no labels.")
        elif found != expected:
            self.problems.append(f"{path}: holds {sorted(found)}, but its JSON file lists {sorted(expected)}.")


def _check_index(root: Path, problems: list[str]) -> None:
    index = root / layout.INDEX_DIR
    missing = [name for name in _INDEX_FILES if not (index / name).is_file()]
    if missing:
        problems.append(f"{index}: has no {', '.join(missing)}; run build_index.")
        return
    built = min((index / name).stat().st_mtime for name in _INDEX_FILES)
    sources = []
    for dataset_folder in root.glob("ds-*"):
        sources += dataset_folder.glob("*.json")
        sources += dataset_folder.glob("*/*.json")
        sources += dataset_folder.glob("*/*/*.json")
    if any(path.stat().st_mtime > built for path in sources):
        problems.append(f"{index}: older than some metadata files; run build_index.")


def validate(
    root: Path,
    publishing: bool = False,
    registry: Mapping[int, str] | str | Path | None = None,
    zips_checked: bool = False,
) -> list[str]:
    """The problems found in a database (see the module docstring).

    registry: the BoneHub registry (see registry.py), as loaded by load_registry or as its URL or path. When given,
    every source of every dataset must be a BoneHub ID of it.
    zips_checked: the model zips were just brought up to date by pack, which compared each with its folder; they
        are not compared again (prepare uses it). A zip that is missing is still a problem.
    """
    root = Path(root)
    checker = _Checker(None if registry is None else as_registry(registry), publishing, zips_checked)
    problems = checker.problems
    for entry in sorted(root.glob("ds-*")):
        if entry.is_dir() and layout.parse_dataset_dir(entry.name) is None:
            problems.append(f"{entry}: not a dataset folder (ds-NNNN).")
    dataset_folders = sorted(p for p in root.iterdir() if p.is_dir() and layout.parse_dataset_dir(p.name) is not None)
    if not dataset_folders:
        return problems + [f"{root}: has no dataset folders (ds-NNNN)."]
    for dataset_folder in dataset_folders:
        if not (dataset_folder / layout.DATASET_JSON).is_file():
            problems.append(f"{dataset_folder}: has no {layout.DATASET_JSON}.")
            continue
        info = checker.load_dataset(dataset_folder)
        if info is not None:
            checker.check_dataset(dataset_folder, info)

    for (dataset_folder, source_id, kind), count in sorted(checker.missing_licenses.items()):
        problems.append(
            f"{dataset_folder}: {count} {kind} file(s) of source {source_id} have no known license; give it in "
            "dataset.json (the source's license, or that of its subset)."
        )
    if publishing:
        _check_index(root, problems)
    return problems
