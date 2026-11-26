"""The metadata files of a BoneHub-format dataset.

    dataset.json                  DatasetInfo: the dataset and the registry datasets it is made from (its sources)
    subject.json                  SubjectInfo: the person, and their ID in each source
    scan.json                     ScanInfo: the scan (modality, age, weight, ...)
    image.json                    FileInfo: where the image came from and its license, where it differs
    segmentation-NN.json, mesh-NN.json, nurbs-NN.json
                                  LabeledFileInfo: the same, and the labels the file holds with their origin

They hold only what cannot be read from the data files themselves. Image geometry, label counts and which bones touch
the image border are computed into the index (see index.py).

A source is a dataset of the BoneHub registry, named by its BoneHub ID (see registry.py). A dataset of a database is
made from one source or more: e.g. the Visible Human dataset holds the CTs of source 153 and the cryosections of
source 109. Every data file names its source; its license is the file's own where its JSON gives one, else that of
the source's subset, else the source's (file_license).
"""

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .labelmap import LabelMap
from .layout import LABELED_KINDS
from .licenses import LICENSE_KINDS, check_license

#: The kinds of per-label data a scan can hold: each file of these kinds maps its labels to their origin.
LABEL_KINDS: tuple[str, ...] = LABELED_KINDS

_LABEL_ORDER = {label.name: i for i, label in enumerate(LabelMap) if label.value}
_ORIGIN = re.compile(r"source|manual|(model|derived):[^\s@:]+@[^\s@]+")
_SCAN_DIR = re.compile(r"scan-\d{2}")


def check_label(label: str) -> str:
    """Return `label` if it is a LabelMap name (other than BACKGROUND), else raise ValueError."""
    if label not in _LABEL_ORDER:
        raise ValueError(f"Invalid label: {label}. Refer to 'bonehub_format/labelmap.py' to see valid labels.")
    return label


def check_origin(origin: str) -> str:
    """Return `origin` if it is valid, else raise ValueError.

    The origin of a label says how its source came by it:
    source                    came with the source dataset
    manual                    drawn by hand
    model:<name>@<version>    predicted by a model, e.g. model:TotalSegmentator@2.4
    derived:<tool>@<version>  computed from another file of the same scan, e.g. a mesh from the segmentation
    """
    if not isinstance(origin, str) or not _ORIGIN.fullmatch(origin):
        raise ValueError(
            f"Invalid origin {origin!r}. An origin is 'source', 'manual', 'model:<name>@<version>' or "
            "'derived:<tool>@<version>'."
        )
    return origin


def check_source_id(value) -> int:
    """Return a source's BoneHub ID as an int (a JSON key such as '153' included), else raise ValueError."""
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if type(value) is not int or value < 1:
        raise ValueError(f"A source is named by its BoneHub ID in the registry, a positive integer, not {value!r}.")
    return value


def sort_labels(labels: Iterable[str]) -> list[str]:
    """The labels in label map order."""
    return sorted(labels, key=lambda label: _LABEL_ORDER[check_label(label)])


def _by_source_id(value: Mapping) -> dict:
    """A mapping keyed by source ID (JSON keys are strings), with int keys in increasing order."""
    if not isinstance(value, Mapping):
        return value  # left to the type check
    keyed = {check_source_id(key): item for key, item in value.items()}
    return {key: keyed[key] for key in sorted(keyed)}


def write_json(path: Path, data) -> None:
    """Write `data` as JSON, replacing the file only once it is completely written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.write("\n")
        os.replace(temp_path, path)
    except BaseException:
        Path(temp_path).unlink(missing_ok=True)
        raise


class _Metadata(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)

    def to_dict(self) -> dict:
        """The fields that are set, in definition order, as they are written to the file."""
        return self.model_dump(mode="json", exclude_none=True)

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path):
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(json.load(f))


class Licenses(_Metadata):
    """The license of each kind of data (see licenses.py); a kind that is not given is not known."""

    image: str | None = Field(None, description="License of the images")
    segmentation: str | None = Field(None, description="License of the segmentations")
    mesh: str | None = Field(None, description="License of the meshes")
    nurbs: str | None = Field(None, description="License of the NURBS surfaces")

    @field_validator(*LICENSE_KINDS)
    @classmethod
    def _check(cls, value):
        return value if value is None else check_license(value)

    @classmethod
    def all(cls, license: str) -> "Licenses":
        """The same license for every kind of data."""
        return cls(**{kind: license for kind in LICENSE_KINDS})

    def of(self, kind: str) -> str | None:
        if kind not in LICENSE_KINDS:
            raise ValueError(f"Unknown kind '{kind}'; the kinds are {list(LICENSE_KINDS)}.")
        return getattr(self, kind)


class SubsetInfo(_Metadata):
    """A collection inside a source that comes from somewhere else and has its own terms, e.g. one of the collections
    of CADS. The files of its scans name it (FileInfo.subset)."""

    name: str = Field(..., description="Name of the collection")
    description: str | None = Field(None, description="What the collection is")
    url: str | None = Field(None, description="Where the original collection is published")
    paper: str | None = Field(None, description="DOI or URL of the paper describing the collection")
    license: Licenses | None = Field(None, description="License of the collection's data, where it differs")
    remarks: str | None = Field(None, description="Remarks about the collection")


class SourceInfo(_Metadata):
    """A dataset of the BoneHub registry that a dataset is made from, and how its data was converted."""

    name: str = Field(..., description="Name of the source dataset")
    description: str | None = Field(None, description="What the source is and how its data was converted")
    url: str | None = Field(None, description="Where the source dataset is published")
    paper: str | None = Field(None, description="DOI or URL of the paper describing the source dataset")
    country: str | None = Field(None, description="Country of origin")
    release_date: str | None = Field(None, description="Release date (or year) of the source dataset")
    version: str | None = Field(None, description="Version of the source dataset that was converted")
    license: Licenses | None = Field(None, description="License of each kind of data the source provides")
    subsets: dict[str, SubsetInfo] | None = Field(
        None, description="Collections inside the source with terms of their own, by the name its files give them"
    )
    annotated_labels: list[str] | None = Field(
        None,
        description="Labels the segmentations of this source cover. In such a segmentation, a covered label that is "
        "missing is absent from the image; a label not covered was not segmented. Empty: nothing is segmented. "
        "None: not known, or it differs between scans",
    )
    remarks: str | None = Field(None, description="Remarks about the source")

    @field_validator("annotated_labels")
    @classmethod
    def _check_annotated_labels(cls, value):
        return value if value is None else sort_labels(set(value))

    def license_of(self, kind: str, subset: str | None = None) -> str | None:
        """The license of a kind of data: the subset's where it gives one, else the source's."""
        if subset is not None and self.subsets and subset in self.subsets:
            own = self.subsets[subset].license
            if own is not None and own.of(kind) is not None:
                return own.of(kind)
        return self.license.of(kind) if self.license else None


class DatasetInfo(_Metadata):
    """dataset.json: a dataset of the database and the registry datasets (sources) it is made from."""

    name: str = Field(..., description="Name of the dataset")
    description: str | None = Field(None, description="How the dataset was made from its sources, where it is not one source")
    sources: dict[int, SourceInfo] = Field(..., min_length=1, description="The sources, by BoneHub ID")
    remarks: str | None = Field(None, description="Remarks about the dataset")
    schema_version: str | None = Field(None, description="Version of the BoneHub format; set by the converter")

    @field_validator("sources", mode="before")
    @classmethod
    def _check_sources(cls, value):
        return _by_source_id(value)

    def source(self, source_id: int) -> SourceInfo | None:
        return self.sources.get(source_id)


class SubjectInfo(_Metadata):
    """subject.json: the person; anything that can change between scans is in scan.json."""

    sources: dict[int, str] = Field(
        default_factory=dict, description="ID of the subject in each source it comes from, by BoneHub ID"
    )
    gender: Literal["M", "F", "O"] | None = Field(None, description="M, F or O (other); None when unknown")
    race: str | None = Field(None, description="Race, in the words of the source, e.g. 'White'; None when unknown")
    ethnicity: str | None = Field(
        None, description="Ethnicity, in the words of the source, e.g. 'Hispanic or Latino'; None when unknown"
    )
    remarks: str | None = Field(None, description="Remarks about the subject")

    @field_validator("sources", mode="before")
    @classmethod
    def _check_sources(cls, value):
        return _by_source_id(value)

    def source_subject_id(self, source_id: int) -> str | None:
        """The subject's ID in a source, or None if it is not in it."""
        return self.sources.get(source_id)


class ScanInfo(_Metadata):
    """scan.json: one image of a subject (or one set of models, for a scan without an image)."""

    modality: str | None = Field(None, description="Imaging modality as a DICOM modality code, e.g. CT or MR")
    age: int | None = Field(None, ge=0, description="Age at the time of the scan, in years")
    weight: float | None = Field(None, gt=0, description="Weight at the time of the scan, in kg")
    height: float | None = Field(None, gt=0, description="Height at the time of the scan, in cm")
    bmi: float | None = Field(None, gt=0, description="Body mass index at the time of the scan")
    postmortem: bool | None = Field(None, description="True for a scan made after death")
    days_from_baseline: int | None = Field(None, description="Days since the subject's first scan, for scan series")
    registered_to: str | None = Field(
        None, description="The scan of the same subject this image is registered to, e.g. scan-02"
    )
    remarks: str | None = Field(None, description="Remarks about the scan")

    @field_validator("registered_to")
    @classmethod
    def _check_registered_to(cls, value):
        if value is not None and not _SCAN_DIR.fullmatch(value):
            raise ValueError(f"registered_to names a scan folder such as 'scan-01', not {value!r}.")
        return value


class FileInfo(_Metadata):
    """image.json: where a data file came from and under which terms."""

    source: int = Field(..., ge=1, description="BoneHub ID of the source the file came from (see dataset.json)")
    source_scan_id: str = Field(..., description="ID of the scan in the source, unique within it, e.g. its relative path")
    subset: str | None = Field(None, description="The collection of the source the scan belongs to")
    source_split: str | None = Field(None, description="Split of the source dataset, e.g. train, val or test")
    license: str | None = Field(None, description="License of the file, where it differs from its source's (or subset's)")
    remarks: str | None = Field(None, description="Remarks about the file")

    @field_validator("license")
    @classmethod
    def _check_license(cls, value):
        return value if value is None else check_license(value)


class LabeledFileInfo(FileInfo):
    """segmentation-NN.json, mesh-NN.json, nurbs-NN.json: a data file that holds labels, and the origin of each."""

    labels: dict[str, str] = Field(default_factory=dict, description="Label -> origin, in label map order")

    @field_validator("labels")
    @classmethod
    def _check_labels(cls, value):
        for origin in value.values():
            check_origin(origin)
        return {label: value[label] for label in sort_labels(value)}

    def set_labels(self, labels: Iterable[str], origin: str | Mapping[str, str]) -> None:
        """Record the labels of the file, with one origin for all or one per label."""
        labels = list(labels)
        if isinstance(origin, str):
            origins = {label: origin for label in labels}
        else:
            missing = [label for label in labels if label not in origin]
            if missing:
                raise ValueError(f"No origin given for {missing}.")
            origins = {label: origin[label] for label in labels}
        self.labels = origins


def file_license(kind: str, info: FileInfo, dataset: DatasetInfo | None) -> str | None:
    """The license of a data file of a kind ('image', 'segmentation', 'mesh', 'nurbs'): the file's own where its JSON
    gives one, else that of its source's subset, else its source's (from `dataset`, its dataset.json). None when no
    license is known."""
    if info.license is not None:
        return info.license
    source = dataset.source(info.source) if dataset is not None else None
    return source.license_of(kind, info.subset) if source is not None else None
