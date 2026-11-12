"""Checking that a database follows the BoneHub format.

    problems = validate(root)                         # the data itself
    problems = validate(root, registry=REGISTRY_URL)  # and that every dataset ID is a BoneHub ID of the registry
    problems = validate(root, publishing=True)        # and that it is ready to upload: licenses, zips and index

Each problem is a line of text naming the file or folder concerned; an empty list means none was found.
"""

from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from . import layout
from ._version import SCHEMA_VERSION, is_compatible_schema_version
from .archive import model_folder_files, zip_is_current
from .metadata import DatasetInfo, ScanInfo, SubjectInfo
from .registry import as_registry
from .segmentation_file import read_segmentation_labels

_INDEX_FILES = ("datasets.parquet", "scans.parquet", "labels.parquet")
_SCAN_ENTRIES = {
    layout.SCAN_JSON,
    layout.IMAGE_FILE,
    layout.SEGMENTATION_FILE,
    *layout.MODEL_KINDS,
    *(f"{kind}.zip" for kind in layout.MODEL_KINDS),
}


def _load(model, path: Path, problems: list[str]):
    try:
        return model.load(path)
    except (ValidationError, ValueError, OSError) as e:
        problems.append(f"{path}: {e}")
        return None


def _model_labels(scan_folder: Path, kind: str, problems: list[str], publishing: bool) -> set[str] | None:
    """The labels a scan has files for, of one model kind; None if they cannot be read."""
    folder = layout.model_folder(scan_folder, kind)
    zip_path = layout.model_zip(scan_folder, kind)
    try:
        if folder.is_dir():
            labels = {f.stem for f in model_folder_files(folder, kind)}
            if labels and zip_path.is_file() and not zip_is_current(folder, zip_path, kind):
                problems.append(f"{zip_path}: does not match {folder.name}/; run pack.")
            elif labels and publishing and not zip_path.is_file():
                problems.append(f"{folder}: has no {zip_path.name}, which is what gets published; run pack.")
            return labels
        if zip_path.is_file():
            import zipfile

            with zipfile.ZipFile(zip_path) as archive:
                return {Path(name).stem for name in archive.namelist()}
    except (ValueError, OSError) as e:
        problems.append(f"{folder}: {e}")
        return None
    return set()


def _check_scan(scan_folder: Path, sibling_names: set[str], problems: list[str], publishing: bool) -> None:
    for entry in scan_folder.iterdir():
        if entry.name not in _SCAN_ENTRIES:
            problems.append(f"{entry}: not part of a scan folder.")
    info = _load(ScanInfo, scan_folder / layout.SCAN_JSON, problems)
    if info is None:
        return
    if info.registered_to is not None and (info.registered_to not in sibling_names or info.registered_to == scan_folder.name):
        problems.append(f"{scan_folder / layout.SCAN_JSON}: registered_to names '{info.registered_to}', not another scan of this subject.")

    segmentation = scan_folder / layout.SEGMENTATION_FILE
    expected = set(info.labels("segmentation"))
    if segmentation.is_file():
        try:
            found = set(read_segmentation_labels(segmentation))
        except (ValueError, OSError) as e:
            problems.append(f"{segmentation}: {e}")
        else:
            if found != expected:
                problems.append(
                    f"{segmentation}: holds {sorted(found)}, but scan.json lists {sorted(expected)} for its segmentation."
                )
    elif expected:
        problems.append(f"{scan_folder}: scan.json lists a segmentation, but there is no {layout.SEGMENTATION_FILE}.")

    for kind in layout.MODEL_KINDS:
        found = _model_labels(scan_folder, kind, problems, publishing)
        expected = set(info.labels(kind))
        if found is not None and found != expected:
            problems.append(f"{scan_folder}: has {kind} files for {sorted(found)}, but scan.json lists {sorted(expected)}.")


def _check_dataset(
    dataset_folder: Path, registry: Mapping[int, str] | None, problems: list[str], publishing: bool
) -> None:
    """Check one dataset folder."""
    dataset_id = layout.parse_dataset_dir(dataset_folder.name)
    dataset_json = dataset_folder / layout.DATASET_JSON
    if registry is not None and dataset_id not in registry:
        problems.append(f"{dataset_folder}: {dataset_id} is not a BoneHub ID of the registry.")
    info = _load(DatasetInfo, dataset_json, problems)
    if info is None:
        return
    if not is_compatible_schema_version(info.schema_version):
        problems.append(
            f"{dataset_json}: written in format version {info.schema_version}, which this package (format version "
            f"{SCHEMA_VERSION}) cannot read."
        )
        return
    if publishing and info.license is None:
        problems.append(f"{dataset_json}: has no license.")
    if dataset_id in (info.derived_from or []):
        problems.append(f"{dataset_json}: derived_from names the dataset itself.")
    for parent in info.derived_from or []:
        if registry is not None and parent not in registry:
            problems.append(f"{dataset_json}: derived_from names {parent}, which is not a BoneHub ID of the registry.")

    subject_folders = []
    for entry in sorted(dataset_folder.iterdir()):
        if entry.name == layout.DATASET_JSON:
            continue
        if layout.parse_subject_dir(entry.name) is None or not entry.is_dir():
            problems.append(f"{entry}: not a subject folder (sub-NNNNNN).")
            continue
        subject_folders.append(entry)

    for subject_folder in subject_folders:
        if _load(SubjectInfo, subject_folder / layout.SUBJECT_JSON, problems) is None:
            continue
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
            _check_scan(scan_folder, names, problems, publishing)


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


def validate(root: Path, publishing: bool = False, registry: Mapping[int, str] | str | Path | None = None) -> list[str]:
    """The problems found in a database (see the module docstring).

    registry: the BoneHub registry (see registry.py), as loaded by load_registry or as its URL or path. When given,
    every dataset ID and every derived_from must be a BoneHub ID of it.
    """
    root = Path(root)
    problems: list[str] = []
    registry = None if registry is None else as_registry(registry)
    for entry in sorted(root.glob("ds-*")):
        if entry.is_dir() and layout.parse_dataset_dir(entry.name) is None:
            problems.append(f"{entry}: not a dataset folder (ds-NNNN, named after the dataset's BoneHub ID).")
    dataset_folders = sorted(p for p in root.iterdir() if p.is_dir() and layout.parse_dataset_dir(p.name) is not None)
    if not dataset_folders:
        return problems + [f"{root}: has no dataset folders (ds-NNNN)."]
    for dataset_folder in dataset_folders:
        if not (dataset_folder / layout.DATASET_JSON).is_file():
            problems.append(f"{dataset_folder}: has no {layout.DATASET_JSON}.")
            continue
        _check_dataset(dataset_folder, registry, problems, publishing)

    if publishing:
        _check_index(root, problems)
    return problems
