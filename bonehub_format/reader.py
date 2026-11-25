"""Reading a BoneHub-format database: its datasets, their subjects and the subjects' scans.

    database = Database("path/to/database")
    for scan in database.scans():
        print(scan.key, scan.info.modality, scan.image_path, scan.license("image"))
        for dataset_id, path in scan.segmentation_paths().items():   # one segmentation per provider dataset
            print(dataset_id, scan.info.labels("segmentation", dataset_id), path)
        for label, files in scan.model_files("mesh", dataset_id=180).items():
            mesh_bytes = files[0].read_bytes()

Metadata is read when first used and then kept.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path

from . import layout
from ._version import SCHEMA_VERSION, is_compatible_schema_version
from .metadata import DatasetInfo, ScanInfo, SubjectInfo, file_license


def _ids(folder: Path, parse, json_name: str) -> list[int]:
    if not folder.is_dir():
        return []
    ids = []
    for entry in folder.iterdir():
        entry_id = parse(entry.name)
        if entry_id is not None and (entry / json_name).is_file():
            ids.append(entry_id)
    return sorted(ids)


class Database:
    """A database: a root folder of BoneHub-format datasets (ds-0001, ds-0027, ...), harmonized under one label map."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"'{self.path}' is not a folder.")
        self._datasets: dict[int, Dataset] = {}

    def dataset_ids(self) -> list[int]:
        return _ids(self.path, layout.parse_dataset_dir, layout.DATASET_JSON)

    def datasets(self) -> list[Dataset]:
        return [self.dataset(dataset_id) for dataset_id in self.dataset_ids()]

    def dataset(self, dataset_id: int) -> Dataset:
        """The dataset folder of this ID (kept, so that its info is read once)."""
        if dataset_id not in self._datasets:
            self._datasets[dataset_id] = Dataset(self, dataset_id)
        return self._datasets[dataset_id]

    def has_dataset(self, dataset_id: int) -> bool:
        return (layout.dataset_path(self.path, dataset_id) / layout.DATASET_JSON).is_file()

    def scans(self) -> Iterator[Scan]:
        for dataset in self.datasets():
            yield from dataset.scans()

    def __repr__(self) -> str:
        return f"Database('{self.path}')"


class Dataset:
    """A dataset folder (ds-NNNN, named after its BoneHub ID). Its info is checked against the format version this package reads, unless
    check_version is False."""

    def __init__(self, database: Database | Path, dataset_id: int, check_version: bool = True):
        self.database = database if isinstance(database, Database) else Database(database)
        self.id = dataset_id
        self.path = layout.dataset_path(self.database.path, dataset_id)
        self.check_version = check_version
        if not (self.path / layout.DATASET_JSON).is_file():
            raise FileNotFoundError(f"'{self.path}' has no {layout.DATASET_JSON}.")

    @cached_property
    def info(self) -> DatasetInfo:
        info = DatasetInfo.load(self.path / layout.DATASET_JSON)
        if self.check_version and not is_compatible_schema_version(info.schema_version):
            raise ValueError(
                f"'{self.path.name}' was written in version {info.schema_version or '(not recorded)'} of the BoneHub "
                f"format, which this package (format version {SCHEMA_VERSION}) cannot read. Convert it again."
            )
        return info

    def subject_ids(self) -> list[int]:
        return _ids(self.path, layout.parse_subject_dir, layout.SUBJECT_JSON)

    def subjects(self) -> list[Subject]:
        return [Subject(self, subject_id) for subject_id in self.subject_ids()]

    def subject(self, subject_id: int) -> Subject:
        return Subject(self, subject_id)

    def scans(self) -> Iterator[Scan]:
        for subject in self.subjects():
            yield from subject.scans()

    def __len__(self) -> int:
        return len(self.subject_ids())

    def __repr__(self) -> str:
        return f"Dataset('{self.path}')"


class Subject:
    """A subject folder (sub-NNNNNN): one person and their scans."""

    def __init__(self, dataset: Dataset, subject_id: int):
        self.dataset = dataset
        self.id = subject_id
        self.path = dataset.path / layout.subject_dir_name(subject_id)
        if not (self.path / layout.SUBJECT_JSON).is_file():
            raise FileNotFoundError(f"'{self.path}' has no {layout.SUBJECT_JSON}.")

    @cached_property
    def info(self) -> SubjectInfo:
        return SubjectInfo.load(self.path / layout.SUBJECT_JSON)

    def scan_ids(self) -> list[int]:
        return _ids(self.path, layout.parse_scan_dir, layout.SCAN_JSON)

    def scans(self) -> list[Scan]:
        return [Scan(self, scan_id) for scan_id in self.scan_ids()]

    def scan(self, scan_id: int) -> Scan:
        return Scan(self, scan_id)

    def __repr__(self) -> str:
        return f"Subject('{self.path}')"


class Scan:
    """A scan folder (scan-NN): one image of a subject and the data derived from it."""

    def __init__(self, subject: Subject, scan_id: int):
        self.subject = subject
        self.id = scan_id
        self.path = subject.path / layout.scan_dir_name(scan_id)
        if not (self.path / layout.SCAN_JSON).is_file():
            raise FileNotFoundError(f"'{self.path}' has no {layout.SCAN_JSON}.")

    @property
    def dataset(self) -> Dataset:
        return self.subject.dataset

    @property
    def key(self) -> str:
        """The scan's path relative to the database's root folder, e.g. 'ds-0027/sub-000001/scan-01'."""
        return layout.scan_key(self.dataset.id, self.subject.id, self.id)

    @cached_property
    def info(self) -> ScanInfo:
        return ScanInfo.load(self.path / layout.SCAN_JSON)

    @property
    def provider_ids(self) -> list[int]:
        """The BoneHub IDs of the datasets that gave the scan data (its home dataset among them if it gave any)."""
        return self.info.provider_ids

    def license(self, kind: str, dataset_id: int | None = None) -> str | None:
        """The license of a kind of data ('image', 'segmentation', 'mesh', 'nurbs') of the scan, given by a provider
        dataset (for the image, the dataset it came from): the scan's own for that provider where scan.json gives one,
        else that of the provider's subset, else the provider dataset's. None if it is not known, e.g. when the
        provider's dataset.json is not in the database."""
        if kind == "image":
            dataset_id = self.info.image_provider
            if dataset_id is None:
                return None
        elif dataset_id is None:
            raise ValueError(f"Say which provider dataset's {kind} the license is wanted for.")
        database = self.dataset.database
        provider_dataset = database.dataset(dataset_id).info if database.has_dataset(dataset_id) else None
        return file_license(self.info, kind, dataset_id, provider_dataset)

    @property
    def image_path(self) -> Path | None:
        path = self.path / layout.IMAGE_FILE
        return path if path.is_file() else None

    def segmentation_path(self, dataset_id: int) -> Path | None:
        """The segmentation a provider dataset gave the scan, or None."""
        path = layout.segmentation_file(self.path, dataset_id)
        return path if path.is_file() else None

    def segmentation_paths(self) -> dict[int, Path]:
        """Provider dataset ID -> its segmentation, for every provider that gave one."""
        paths = {dataset_id: self.segmentation_path(dataset_id) for dataset_id in self.provider_ids}
        return {dataset_id: path for dataset_id, path in paths.items() if path is not None}

    def model_files(self, kind: str, dataset_id: int) -> dict[str, list[Path | zipfile.Path]]:
        """Label -> files of a model kind ('mesh' or 'nurbs') a provider dataset gave the scan, from the folder if the
        scan has it, else from the zip.

        Files from a zip are zipfile.Path objects, read with .read_bytes() or .open(); unpack the zips (see
        archive.unpack) to have plain files instead.
        """
        suffixes = layout.MODEL_KINDS.get(kind)
        if suffixes is None:
            raise ValueError(f"Unknown kind '{kind}'; the kinds are {list(layout.MODEL_KINDS)}.")
        folder = layout.model_folder(self.path, kind, dataset_id)
        zip_path = layout.model_zip(self.path, kind, dataset_id)
        if folder.is_dir():
            entries = [entry for entry in folder.iterdir() if entry.is_file()]
        elif zip_path.is_file():
            entries = [entry for entry in zipfile.Path(zip_path).iterdir() if entry.is_file()]
        else:
            return {}
        files: dict[str, list[Path | zipfile.Path]] = {}
        for entry in sorted(entries, key=lambda entry: entry.name):
            stem, suffix = Path(entry.name).stem, Path(entry.name).suffix.lower()
            if suffix in suffixes:
                files.setdefault(stem, []).append(entry)
        return files

    def __repr__(self) -> str:
        return f"Scan('{self.path}')"
