"""The metadata files of a BoneHub-format dataset: dataset.json, subject.json and scan.json.

They hold only what cannot be read from the data files themselves. Image geometry, label counts and which bones touch
the image border are computed into the index (see index.py).

A scan's data can come from several datasets, its providers, which are named by their provider key ('ds-0040', see
layout.provider_key). A subject lives in one dataset, its home, and lists its ID in every provider; a scan says which
provider its image came from and, per provider, which labels it gave and under which license.
"""

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import layout
from .labelmap import LabelMap
from .layout import MODEL_KINDS
from .licenses import LICENSE_KINDS, check_license

#: The kinds of per-label data a scan can hold, as named in scan.json: each maps a label to its origin.
LABEL_KINDS: tuple[str, ...] = ("segmentation", *MODEL_KINDS)

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

    The origin of a label says how its provider dataset came by it:
    source                    came with the provider's source dataset
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


def check_provider_key(key: str) -> str:
    """Return `key` if it names a dataset such as 'ds-0040', else raise ValueError."""
    if not isinstance(key, str) or layout.parse_provider_key(key) is None:
        raise ValueError(f"A provider is named by its dataset folder, e.g. 'ds-0040', not {key!r}.")
    return key


def sort_labels(labels: Iterable[str]) -> list[str]:
    """The labels in label map order."""
    return sorted(labels, key=lambda label: _LABEL_ORDER[check_label(label)])


def _sorted_providers(value: Mapping[str, object]) -> dict:
    for key in value:
        check_provider_key(key)
    return {key: value[key] for key in sorted(value)}


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
    """A collection inside a dataset that comes from somewhere else and has its own terms, e.g. one of the source
    collections of CADS. Its scans name it in scan.json (ProviderInfo.subset)."""

    name: str = Field(..., description="Name of the collection")
    description: str | None = Field(None, description="What the collection is")
    url: str | None = Field(None, description="Where the original collection is published")
    paper: str | None = Field(None, description="DOI or URL of the paper describing the collection")
    citation: str | None = Field(None, description="BibTeX entry to cite the collection with")
    license: Licenses | None = Field(None, description="License of the collection's data, where it differs")
    remarks: str | None = Field(None, description="Remarks about the collection")


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
    license: Licenses | None = Field(None, description="License of each kind of data the dataset provides")
    subsets: dict[str, SubsetInfo] | None = Field(
        None, description="Collections inside the dataset with terms of their own, by the name scans give them"
    )
    derived_from: list[int] | None = Field(
        None,
        description="BoneHub IDs of the datasets this one is made from, wholly or in part, e.g. the dataset a newly "
        "segmented copy was made from. Where a scan was made from a scan of a BoneHub-format dataset, its "
        "source_scan_id is that scan's path, e.g. ds-0027/sub-000001/scan-01",
    )
    annotated_labels: list[str] | None = Field(
        None,
        description="Labels the segmentations this dataset provides cover. In such a segmentation, a covered label "
        "that is missing is absent from the image; a label not covered was not segmented. Empty: nothing is "
        "segmented. None: not known, or it differs between scans",
    )
    remarks: str | None = Field(None, description="Remarks about the dataset")
    schema_version: str | None = Field(None, description="Version of the BoneHub format; set by the converter")

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

    def license_of(self, kind: str, subset: str | None = None) -> str | None:
        """The license of a kind of data: the subset's where it gives one, else the dataset's."""
        if subset is not None and self.subsets and subset in self.subsets:
            own = self.subsets[subset].license
            if own is not None and own.of(kind) is not None:
                return own.of(kind)
        return self.license.of(kind) if self.license else None


class SubjectInfo(_Metadata):
    """subject.json: the person; anything that can change between scans is in scan.json."""

    sources: dict[str, str] = Field(
        default_factory=dict,
        description="ID of the subject in each provider dataset, by provider key; the home dataset is one of them",
    )
    gender: Literal["M", "F", "O"] | None = Field(None, description="M, F or O (other); None when unknown")
    remarks: str | None = Field(None, description="Remarks about the subject")

    @field_validator("sources")
    @classmethod
    def _check_sources(cls, value):
        return _sorted_providers(value)

    def source_subject_id(self, dataset_id: int) -> str | None:
        """The subject's ID in a provider dataset, or None if it is not in it."""
        return self.sources.get(layout.provider_key(dataset_id))


class ProviderInfo(_Metadata):
    """What one provider dataset gave a scan, in scan.json."""

    source_scan_id: str = Field(
        ..., description="ID of the scan in the provider's source dataset, unique within it, e.g. its relative path"
    )
    subset: str | None = Field(None, description="The collection of the provider dataset the scan belongs to")
    source_split: str | None = Field(None, description="Split of the source dataset, e.g. train, val or test")
    license: Licenses | None = Field(None, description="License of this scan's data, where it differs from the provider's")
    segmentation: dict[str, str] | None = Field(None, description="Label -> origin, for segmentation.<provider>.seg.nrrd")
    mesh: dict[str, str] | None = Field(None, description="Label -> origin, for mesh.<provider>/ (mesh.<provider>.zip)")
    nurbs: dict[str, str] | None = Field(None, description="Label -> origin, for nurbs.<provider>/ (nurbs.<provider>.zip)")
    remarks: str | None = Field(None, description="Remarks about this provider's data of the scan")

    @field_validator(*LABEL_KINDS)
    @classmethod
    def _check_origins(cls, value):
        if value is None:
            return None
        for origin in value.values():
            check_origin(origin)
        return {label: value[label] for label in sort_labels(value)}

    def labels(self, kind: str) -> list[str]:
        """The labels this provider gave data of this kind for, in label map order."""
        if kind not in LABEL_KINDS:
            raise ValueError(f"Unknown kind '{kind}'; the kinds are {list(LABEL_KINDS)}.")
        return list(getattr(self, kind) or {})

    def set_labels(self, kind: str, labels: Iterable[str], origin: str | Mapping[str, str]) -> None:
        """Record the labels this provider gave data of this kind for, with one origin for all or one per label."""
        if kind not in LABEL_KINDS:
            raise ValueError(f"Unknown kind '{kind}'; the kinds are {list(LABEL_KINDS)}.")
        labels = list(labels)
        if isinstance(origin, str):
            origins = {label: origin for label in labels}
        else:
            missing = [label for label in labels if label not in origin]
            if missing:
                raise ValueError(f"No origin given for the {kind} of {missing}.")
            origins = {label: origin[label] for label in labels}
        setattr(self, kind, origins or None)


class ScanInfo(_Metadata):
    """scan.json: one image of a subject and the data derived from it, from one or more provider datasets."""

    image: str | None = Field(None, description="Provider key of the dataset image.nii.gz came from; None: no image")
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
    providers: dict[str, ProviderInfo] = Field(
        default_factory=dict, description="What each provider dataset gave the scan, by provider key"
    )
    remarks: str | None = Field(None, description="Remarks about the scan")

    @field_validator("image")
    @classmethod
    def _check_image(cls, value):
        return value if value is None else check_provider_key(value)

    @field_validator("registered_to")
    @classmethod
    def _check_registered_to(cls, value):
        if value is not None and not _SCAN_DIR.fullmatch(value):
            raise ValueError(f"registered_to names a scan folder such as 'scan-01', not {value!r}.")
        return value

    @field_validator("providers")
    @classmethod
    def _check_providers(cls, value):
        return _sorted_providers(value)

    @model_validator(mode="after")
    def _image_has_a_provider(self):
        if self.image is not None and self.providers and self.image not in self.providers:
            raise ValueError(f"The image comes from {self.image}, which is not one of the providers {list(self.providers)}.")
        return self

    @property
    def provider_ids(self) -> list[int]:
        """The BoneHub IDs of the provider datasets, in order."""
        return [layout.parse_provider_key(key) for key in self.providers]

    @property
    def image_provider(self) -> int | None:
        """The BoneHub ID of the dataset the image came from, or None for a scan without an image."""
        return None if self.image is None else layout.parse_provider_key(self.image)

    def provider(self, dataset_id: int) -> ProviderInfo | None:
        return self.providers.get(layout.provider_key(dataset_id))

    def labels(self, kind: str, dataset_id: int | None = None) -> list[str]:
        """The labels that have data of this kind, from one provider or (by default) from any, in label map order."""
        if dataset_id is not None:
            provider = self.provider(dataset_id)
            return provider.labels(kind) if provider else []
        if kind not in LABEL_KINDS:
            raise ValueError(f"Unknown kind '{kind}'; the kinds are {list(LABEL_KINDS)}.")
        return sort_labels({label for provider in self.providers.values() for label in provider.labels(kind)})

    def source_scan_id(self) -> str | None:
        """The scan's ID in the dataset its image came from, or else in its first provider."""
        if self.image is not None and self.image in self.providers:
            return self.providers[self.image].source_scan_id
        return next(iter(self.providers.values())).source_scan_id if self.providers else None


def file_license(scan: ScanInfo, kind: str, dataset_id: int, provider_dataset: DatasetInfo | None) -> str | None:
    """The license of a kind of data a provider dataset gave a scan: the scan's own for that provider where scan.json
    gives one, else that of the provider's subset, else the provider dataset's (`provider_dataset`, its dataset.json;
    None if it is not known). None when no license is known."""
    provider = scan.provider(dataset_id)
    if provider is None:
        return None
    if provider.license is not None and provider.license.of(kind) is not None:
        return provider.license.of(kind)
    return provider_dataset.license_of(kind, provider.subset) if provider_dataset is not None else None
