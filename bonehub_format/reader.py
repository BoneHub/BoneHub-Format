"""Reading a BoneHub-format root folder: its datasets, their subjects and the subjects' scans.

    root = Root("Z:/BoneHub/BoneHub_Dataset")
    for scan in root.scans():
        print(scan.key, scan.info.modality, scan.info.labels("segmentation"), scan.image_path)
        for label, files in scan.model_files("mesh").items():
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
from .metadata import DatasetInfo, ScanInfo, SubjectInfo


def _ids(folder: Path, parse, json_name: str) -> list[int]:
    if not folder.is_dir():
        return []
    ids = []
    for entry in folder.iterdir():
        entry_id = parse(entry.name)
        if entry_id is not None and (entry / json_name).is_file():
            ids.append(entry_id)
    return sorted(ids)


class Root:
    """A folder of BoneHub-format datasets (ds-001, ds-002, ...)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"'{self.path}' is not a folder.")

    def dataset_ids(self) -> list[int]:
        return _ids(self.path, layout.parse_dataset_dir, layout.DATASET_JSON)

    def datasets(self) -> list[Dataset]:
        return [Dataset(self, dataset_id) for dataset_id in self.dataset_ids()]

    def dataset(self, dataset_id: int) -> Dataset:
        return Dataset(self, dataset_id)

    def scans(self) -> Iterator[Scan]:
        for dataset in self.datasets():
            yield from dataset.scans()

    def __repr__(self) -> str:
        return f"Root('{self.path}')"


class Dataset:
    """A dataset folder (ds-NNN). Its info is checked against the format version this package reads, unless
    check_version is False."""

    def __init__(self, root: Root | Path, dataset_id: int, check_version: bool = True):
        self.root = root if isinstance(root, Root) else Root(root)
        self.id = dataset_id
        self.path = layout.dataset_path(self.root.path, dataset_id)
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
        """The scan's path relative to the root, e.g. 'ds-001/sub-000001/scan-01'."""
        return layout.scan_key(self.dataset.id, self.subject.id, self.id)

    @cached_property
    def info(self) -> ScanInfo:
        return ScanInfo.load(self.path / layout.SCAN_JSON)

    @property
    def license(self) -> str | None:
        """The scan's license: its own where it has one, else its dataset's."""
        return self.info.license or self.dataset.info.license

    @property
    def image_path(self) -> Path | None:
        path = self.path / layout.IMAGE_FILE
        return path if path.is_file() else None

    @property
    def segmentation_path(self) -> Path | None:
        path = self.path / layout.SEGMENTATION_FILE
        return path if path.is_file() else None

    def model_files(self, kind: str) -> dict[str, list[Path | zipfile.Path]]:
        """Label -> files of a model kind ('mesh' or 'nurbs'), from the folder if the scan has it, else from the zip.

        Files from a zip are zipfile.Path objects, read with .read_bytes() or .open(); unpack the zips (see
        archive.unpack) to have plain files instead.
        """
        suffixes = layout.MODEL_KINDS.get(kind)
        if suffixes is None:
            raise ValueError(f"Unknown kind '{kind}'; the kinds are {list(layout.MODEL_KINDS)}.")
        folder = layout.model_folder(self.path, kind)
        zip_path = layout.model_zip(self.path, kind)
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
