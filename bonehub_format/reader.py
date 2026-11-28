"""Reading a BoneHub-format database: its datasets, their subjects and the subjects' scans.

    database = Database("path/to/database")
    for scan in database.scans():
        print(scan.key, scan.info.modality, scan.image_path)
        for segmentation in scan.files("segmentation"):     # segmentation-01.nrrd, segmentation-02.nrrd, ...
            print(segmentation.path, segmentation.source, segmentation.labels, segmentation.license())
        for mesh in scan.files("mesh"):
            for label, files in mesh.model_files().items():
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
from .landmarks import read_landmarks
from .metadata import DatasetInfo, FileInfo, LabeledFileInfo, ScanInfo, SubjectInfo, file_license, sort_labels


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
    """A database: a root folder of BoneHub-format datasets (ds-0001, ds-0002, ...), harmonized under one label map."""

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
    """A dataset folder (ds-NNNN, numbered within the database). Its info is checked against the format version this
    package reads, unless check_version is False."""

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


class DataFile:
    """A data file of a scan and its JSON file: the image (number None), or segmentation-NN, mesh-NN, nurbs-NN or
    landmarks-NN."""

    def __init__(self, scan: Scan, kind: str, number: int | None = None):
        self.scan = scan
        self.kind = kind
        self.number = number
        self.path = layout.data_file(scan.path, kind, number)  # what is published: image.nii.gz, segmentation-01.nrrd, mesh-01.zip
        self.json_path = layout.data_json(scan.path, kind, number)

    @cached_property
    def info(self) -> FileInfo | LabeledFileInfo:
        return (FileInfo if self.kind == "image" else LabeledFileInfo).load(self.json_path)

    @property
    def source(self) -> int:
        """The BoneHub ID of the source the file came from."""
        return self.info.source

    @property
    def labels(self) -> list[str]:
        """The labels the file holds, in label map order (none for the image)."""
        return list(getattr(self.info, "labels", {}))

    @property
    def folder(self) -> Path | None:
        """The unpacked folder of a model or landmarks file (mesh-01/, landmarks-01/), or None."""
        if self.kind not in layout.FOLDER_KINDS:
            return None
        folder = layout.model_folder(self.scan.path, self.kind, self.number)
        return folder if folder.is_dir() else None

    def license(self) -> str | None:
        """The file's license: its own where its JSON gives one, else its source's subset's, else its source's. None
        if it is not known."""
        return file_license(self.kind, self.info, self.scan.dataset.info)

    def model_files(self) -> dict[str, list[Path | zipfile.Path]]:
        """Label -> files of a model file (mesh or NURBS), from its folder if the scan has it, else from the zip.

        Files from a zip are zipfile.Path objects, read with .read_bytes() or .open(); unpack the zips (see
        archive.unpack) to have plain files instead.
        """
        suffixes = layout.MODEL_KINDS.get(self.kind)
        if suffixes is None:
            raise ValueError(f"A {self.kind} file has no model files.")
        folder = self.folder
        if folder is not None:
            entries = [entry for entry in folder.iterdir() if entry.is_file()]
        elif self.path.is_file():
            entries = [entry for entry in zipfile.Path(self.path).iterdir() if entry.is_file()]
        else:
            return {}
        files: dict[str, list[Path | zipfile.Path]] = {}
        for entry in sorted(entries, key=lambda entry: entry.name):
            stem, suffix = Path(entry.name).stem, Path(entry.name).suffix.lower()
            if suffix in suffixes:
                files.setdefault(stem, []).append(entry)
        return files

    def landmarks(self) -> list[dict]:
        """The entries of a landmarks file (see landmarks.py), from its folder if the scan has it, else from the zip."""
        if self.kind != "landmarks":
            raise ValueError(f"A {self.kind} file holds no landmarks.")
        source = self.folder or self.path
        return read_landmarks(source)["landmarks"] if source.exists() else []

    def __repr__(self) -> str:
        return f"DataFile('{self.path}')"


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
        """The scan's path relative to the database's root folder, e.g. 'ds-0001/sub-000001/scan-01'."""
        return layout.scan_key(self.dataset.id, self.subject.id, self.id)

    @cached_property
    def info(self) -> ScanInfo:
        return ScanInfo.load(self.path / layout.SCAN_JSON)

    @property
    def image_path(self) -> Path | None:
        path = self.path / layout.IMAGE_FILE
        return path if path.is_file() else None

    @property
    def image(self) -> DataFile | None:
        """The image and its JSON file, or None for a scan without an image."""
        return DataFile(self, "image") if layout.data_json(self.path, "image").is_file() else None

    def files(self, kind: str | None = None) -> list[DataFile]:
        """The scan's data files of a kind ('image', 'segmentation', 'mesh', 'nurbs', 'landmarks'), or of every kind, in order:
        those that have a JSON file."""
        kinds = layout.DATA_KINDS if kind is None else [layout._check_kind(kind)]
        found = []
        for entry in self.path.iterdir():
            parsed = layout.parse_scan_entry(entry.name)
            if parsed is not None and parsed[2] == "json" and parsed[0] in kinds:
                found.append((layout.DATA_KINDS.index(parsed[0]), parsed[1] or 0, parsed[0]))
        return [DataFile(self, kind, number or None) for _, number, kind in sorted(found)]

    def labels(self, kind: str) -> list[str]:
        """The labels that have data of a kind ('segmentation', 'mesh', 'nurbs', 'landmarks') in any file, in label map
        order."""
        return sort_labels({label for data in self.files(kind) for label in data.labels})

    @property
    def sources(self) -> list[int]:
        """The BoneHub IDs of the sources the scan's files came from."""
        return sorted({data.source for data in self.files()})

    def __repr__(self) -> str:
        return f"Scan('{self.path}')"
