"""Checking that a database follows the BoneHub format.

    problems = validate(root)                         # the data itself
    problems = validate(root, registry=REGISTRY_URL)  # and that every dataset ID is a BoneHub ID of the registry
    problems = validate(root, publishing=True)        # and that it is ready to upload: licenses, zips and index

Each problem is a line of text naming the file or folder concerned; an empty list means none was found.
"""

import zipfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from . import layout
from ._version import SCHEMA_VERSION, is_compatible_schema_version
from .archive import model_folder_files, zip_is_current
from .metadata import DatasetInfo, ScanInfo, SubjectInfo, file_license
from .registry import as_registry
from .segmentation_file import read_segmentation_labels

_INDEX_FILES = ("datasets.parquet", "scans.parquet", "labels.parquet")


def _load(model, path: Path, problems: list[str]):
    try:
        return model.load(path)
    except (ValidationError, ValueError, OSError) as e:
        problems.append(f"{path}: {e}")
        return None


def _model_labels(scan_folder: Path, kind: str, dataset_id: int, problems: list[str], publishing: bool) -> set[str] | None:
    """The labels a scan has files for, of one model kind and provider; None if they cannot be read."""
    folder = layout.model_folder(scan_folder, kind, dataset_id)
    zip_path = layout.model_zip(scan_folder, kind, dataset_id)
    try:
        if folder.is_dir():
            labels = {f.stem for f in model_folder_files(folder, kind)}
            if labels and zip_path.is_file() and not zip_is_current(folder, zip_path, kind):
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
    def __init__(self, root: Path, registry: Mapping[int, str] | None, publishing: bool):
        self.root = root
        self.registry = registry
        self.publishing = publishing
        self.problems: list[str] = []
        self.datasets: dict[int, DatasetInfo] = {}  # the dataset.json of every dataset folder that has a valid one
        self.used_providers: set[int] = set()
        self.missing_licenses: Counter = Counter()  # (provider, kind) -> files without a known license

    def check_scan(self, scan_folder: Path, sibling_names: set[str]) -> None:
        problems = self.problems
        info = _load(ScanInfo, scan_folder / layout.SCAN_JSON, problems)
        if info is None:
            return
        scan_json = scan_folder / layout.SCAN_JSON
        if not info.providers:
            problems.append(f"{scan_json}: lists no providers.")
        if info.registered_to is not None and (info.registered_to not in sibling_names or info.registered_to == scan_folder.name):
            problems.append(f"{scan_json}: registered_to names '{info.registered_to}', not another scan of this subject.")

        has_image = (scan_folder / layout.IMAGE_FILE).is_file()
        if info.image is not None and not has_image:
            problems.append(f"{scan_json}: says the image comes from {info.image}, but there is no {layout.IMAGE_FILE}.")
        elif has_image and info.image is None:
            problems.append(f"{scan_folder / layout.IMAGE_FILE}: scan.json does not say which provider it comes from.")

        for key, provider in info.providers.items():
            dataset_id = layout.parse_provider_key(key)
            self.used_providers.add(dataset_id)
            dataset = self.datasets.get(dataset_id)
            if dataset is None:
                problems.append(f"{scan_json}: provider {key} has no dataset folder with a valid {layout.DATASET_JSON} here.")
            elif provider.subset is not None and provider.subset not in (dataset.subsets or {}):
                problems.append(f"{scan_json}: provider {key} names subset '{provider.subset}', which its dataset.json lacks.")

        entries = {}
        for entry in scan_folder.iterdir():
            if entry.name in (layout.SCAN_JSON, layout.IMAGE_FILE):
                continue
            parsed = layout.parse_scan_entry(entry.name)
            if parsed is None:
                problems.append(f"{entry}: not part of a scan folder.")
                continue
            kind, dataset_id, _ = parsed
            entries.setdefault((kind, dataset_id), []).append(entry)
            if layout.provider_key(dataset_id) not in info.providers:
                problems.append(f"{entry}: comes from {layout.provider_key(dataset_id)}, which scan.json does not list.")

        for key, provider in info.providers.items():
            dataset_id = layout.parse_provider_key(key)
            self._check_segmentation(scan_folder, dataset_id, set(provider.labels("segmentation")))
            for kind in layout.MODEL_KINDS:
                found = _model_labels(scan_folder, kind, dataset_id, problems, self.publishing)
                expected = set(provider.labels(kind))
                if found is not None and found != expected:
                    problems.append(
                        f"{scan_folder}: has {kind} files of {key} for {sorted(found)}, but scan.json lists {sorted(expected)}."
                    )

        if self.publishing:
            kinds = [("image", info.image_provider)] if info.image is not None else []
            for key, provider in info.providers.items():
                kinds += [(kind, layout.parse_provider_key(key)) for kind in ("segmentation", *layout.MODEL_KINDS) if provider.labels(kind)]
            for kind, dataset_id in kinds:
                if file_license(info, kind, dataset_id, self.datasets.get(dataset_id)) is None:
                    self.missing_licenses[(dataset_id, kind)] += 1

    def _check_segmentation(self, scan_folder: Path, dataset_id: int, expected: set[str]) -> None:
        segmentation = layout.segmentation_file(scan_folder, dataset_id)
        if segmentation.is_file():
            try:
                found = set(read_segmentation_labels(segmentation))
            except (ValueError, OSError) as e:
                self.problems.append(f"{segmentation}: {e}")
            else:
                if found != expected:
                    self.problems.append(
                        f"{segmentation}: holds {sorted(found)}, but scan.json lists {sorted(expected)} for it."
                    )
        elif expected:
            self.problems.append(f"{scan_folder}: scan.json lists a segmentation of {layout.provider_key(dataset_id)}, but there is no {segmentation.name}.")

    def load_dataset(self, dataset_folder: Path) -> None:
        """Read and check a dataset.json; keep it for the checks of the scans."""
        problems = self.problems
        dataset_id = layout.parse_dataset_dir(dataset_folder.name)
        dataset_json = dataset_folder / layout.DATASET_JSON
        if self.registry is not None and dataset_id not in self.registry:
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
        if dataset_id in (info.derived_from or []):
            problems.append(f"{dataset_json}: derived_from names the dataset itself.")
        for parent in info.derived_from or []:
            if self.registry is not None and parent not in self.registry:
                problems.append(f"{dataset_json}: derived_from names {parent}, which is not a BoneHub ID of the registry.")
        self.datasets[dataset_id] = info

    def check_dataset(self, dataset_folder: Path) -> int:
        """Check the subjects of a dataset folder; return how many there are."""
        problems = self.problems
        subject_folders = []
        for entry in sorted(dataset_folder.iterdir()):
            if entry.name == layout.DATASET_JSON:
                continue
            if layout.parse_subject_dir(entry.name) is None or not entry.is_dir():
                problems.append(f"{entry}: not a subject folder (sub-NNNNNN).")
                continue
            subject_folders.append(entry)

        for subject_folder in subject_folders:
            subject = _load(SubjectInfo, subject_folder / layout.SUBJECT_JSON, problems)
            if subject is None:
                continue
            if dataset_folder.name not in subject.sources:
                problems.append(f"{subject_folder / layout.SUBJECT_JSON}: sources lacks the subject's ID in {dataset_folder.name}, its home.")
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
                self.check_scan(scan_folder, names)
        return len(subject_folders)


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
    every dataset ID and every derived_from must be a BoneHub ID of it (and so every provider, which must be a
    dataset folder of the database).
    """
    root = Path(root)
    checker = _Checker(root, None if registry is None else as_registry(registry), publishing)
    problems = checker.problems
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
        checker.load_dataset(dataset_folder)

    without_subjects = []
    for dataset_folder in dataset_folders:
        dataset_id = layout.parse_dataset_dir(dataset_folder.name)
        if dataset_id in checker.datasets and checker.check_dataset(dataset_folder) == 0:
            without_subjects.append(dataset_folder)
    for dataset_folder in without_subjects:
        if layout.parse_dataset_dir(dataset_folder.name) not in checker.used_providers:
            problems.append(f"{dataset_folder}: has no subjects and gives data to no scan of the database.")

    for (dataset_id, kind), count in sorted(checker.missing_licenses.items()):
        problems.append(
            f"{layout.provider_key(dataset_id)}: {count} {kind} file(s) it provides have no known license; give it in "
            "its dataset.json (license, or the license of its subset)."
        )
    if publishing:
        _check_index(root, problems)
    return problems
