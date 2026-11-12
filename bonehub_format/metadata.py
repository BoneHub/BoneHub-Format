"""The metadata files of a BoneHub-format dataset: dataset.json, subject.json and scan.json.

They hold only what cannot be read from the data files themselves. Image geometry, label counts and which bones touch
the image border are computed into the index (see index.py).
"""

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .labelmap import BoneLabelMap
from .layout import MODEL_KINDS
from .licenses import check_license

#: The kinds of per-label data a scan can hold, as named in scan.json: each maps a label to its origin.
LABEL_KINDS: tuple[str, ...] = ("segmentation", *MODEL_KINDS)

_LABEL_ORDER = {label.name: i for i, label in enumerate(BoneLabelMap) if label.value}
_ORIGIN = re.compile(r"source|manual|(model|derived):[^\s@:]+@[^\s@]+")
_SCAN_DIR = re.compile(r"scan-\d{2}")


def check_label(label: str) -> str:
    """Return `label` if it is a BoneLabelMap name (other than BACKGROUND), else raise ValueError."""
    if label not in _LABEL_ORDER:
        raise ValueError(f"Invalid label: {label}. Refer to 'bonehub_format/labelmap.py' to see valid labels.")
    return label


def check_origin(origin: str) -> str:
    """Return `origin` if it is valid, else raise ValueError.

    source                    came with the source dataset
    manual                    drawn by hand for this dataset
    model:<name>@<version>    predicted by a model, e.g. model:TotalSegmentator@2.4
    derived:<tool>@<version>  computed from another file of the same scan, e.g. a mesh from the segmentation
    """
    if not isinstance(origin, str) or not _ORIGIN.fullmatch(origin):
        raise ValueError(
            f"Invalid origin {origin!r}. An origin is 'source', 'manual', 'model:<name>@<version>' or "
            "'derived:<tool>@<version>'."
        )
    return origin


def sort_labels(labels: Iterable[str]) -> list[str]:
    """The labels in label map order."""
    return sorted(labels, key=lambda label: _LABEL_ORDER[check_label(label)])


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


class DatasetInfo(_Metadata):
    """dataset.json: the source dataset and how it was converted."""

    name: str = Field(..., description="Name of the dataset")
    description: str | None = Field(None, description="Description of the dataset and of how it was converted")
    url: str | None = Field(None, description="Where the source dataset is published")
    paper: str | None = Field(None, description="DOI or URL of the paper describing the source dataset")
    citation: str | None = Field(None, description="BibTeX entry to cite the source dataset with")
    country: str | None = Field(None, description="Country of origin")
    release_date: str | None = Field(None, description="Release date (or year) of the source dataset")
    version: str | None = Field(None, description="Version of the source dataset that was converted")
    license: str | None = Field(
        None, description="SPDX identifier of the license, or 'restricted' for a custom license with restricted access"
    )
    derived_from: list[int] | None = Field(
        None,
        description="BoneHub IDs of the datasets this one is made from, wholly or in part, and so shares subjects "
        "with, e.g. the dataset a newly segmented copy was made from. It is the only record of which datasets hold the "
        "same people. Where a scan was made from a scan of a BoneHub-format dataset, its source_scan_id is that "
        "scan's path, e.g. ds-0027/sub-000001/scan-01",
    )
    annotated_labels: list[str] | None = Field(
        None,
        description="Labels the segmentations of this dataset cover. In a scan with a segmentation, a covered label "
        "that is missing is absent from the image; a label not covered was not segmented. Empty: nothing is "
        "segmented. None: not known, or it differs between scans",
    )
    remarks: str | None = Field(None, description="Remarks about the dataset")
    schema_version: str | None = Field(None, description="Version of the BoneHub format; set by the converter")

    @field_validator("license")
    @classmethod
    def _check_license(cls, value):
        return value if value is None else check_license(value)

    @field_validator("derived_from")
    @classmethod
    def _check_derived_from(cls, value):
        if value is not None and any(dataset_id < 1 for dataset_id in value):
            raise ValueError("Dataset IDs start at 1.")
        return value

    @field_validator("annotated_labels")
    @classmethod
    def _check_annotated_labels(cls, value):
        return value if value is None else sort_labels(set(value))


class SubjectInfo(_Metadata):
    """subject.json: the person; anything that can change between scans is in scan.json."""

    source_subject_id: str = Field(..., description="ID of the subject in the source dataset")
    gender: Literal["M", "F", "O"] | None = Field(None, description="M, F or O (other); None when unknown")
    remarks: str | None = Field(None, description="Remarks about the subject")


class ScanInfo(_Metadata):
    """scan.json: one image of a subject and the data derived from it."""

    source_scan_id: str = Field(
        ..., description="ID of the scan in the source dataset, unique within the dataset, e.g. its relative path"
    )
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
    source_split: str | None = Field(None, description="Split of the source dataset, e.g. train, val or test")
    license: str | None = Field(None, description="License of this scan, where it differs from the dataset's")
    segmentation: dict[str, str] | None = Field(None, description="Label -> origin, for segmentation.seg.nrrd")
    mesh: dict[str, str] | None = Field(None, description="Label -> origin, for the files in mesh/ (mesh.zip)")
    nurbs: dict[str, str] | None = Field(None, description="Label -> origin, for the files in nurbs/ (nurbs.zip)")
    remarks: str | None = Field(None, description="Remarks about the scan")

    @field_validator("license")
    @classmethod
    def _check_license(cls, value):
        return value if value is None else check_license(value)

    @field_validator("registered_to")
    @classmethod
    def _check_registered_to(cls, value):
        if value is not None and not _SCAN_DIR.fullmatch(value):
            raise ValueError(f"registered_to names a scan folder such as 'scan-01', not {value!r}.")
        return value

    @field_validator(*LABEL_KINDS)
    @classmethod
    def _check_origins(cls, value):
        if value is None:
            return None
        for origin in value.values():
            check_origin(origin)
        return {label: value[label] for label in sort_labels(value)}

    def labels(self, kind: str) -> list[str]:
        """The labels that have data of this kind, in label map order."""
        if kind not in LABEL_KINDS:
            raise ValueError(f"Unknown kind '{kind}'; the kinds are {list(LABEL_KINDS)}.")
        return list(getattr(self, kind) or {})

    def set_labels(self, kind: str, labels: Iterable[str], origin: str | Mapping[str, str]) -> None:
        """Record the labels that have data of this kind, with one origin for all or an origin per label."""
        if kind not in LABEL_KINDS:
            raise ValueError(f"Unknown kind '{kind}'; the kinds are {list(LABEL_KINDS)}.")
        if isinstance(origin, str):
            origins = {label: origin for label in labels}
        else:
            missing = [label for label in labels if label not in origin]
            if missing:
                raise ValueError(f"No origin given for the {kind} of {missing}.")
            origins = {label: origin[label] for label in labels}
        setattr(self, kind, origins or None)
