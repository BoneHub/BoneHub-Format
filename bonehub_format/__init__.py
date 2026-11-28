"""BoneHub-Format: the BoneHub data format for 3D bone datasets, and the tools to build, check and publish harmonized
databases of them.

The base install reads, checks and packs databases; it needs only pydantic. Changing segmentations needs the io
extra (pip install "bonehub-format[io]"). Converting sources, building the index and writing the Hugging Face files and
webpage need the converter extra
(pip install "bonehub-format[converter]") and are imported from their modules:

    bonehub_format.converter   DatasetConverter, SubjectSource, ScanSource
    bonehub_format.export      helpers for converters: images, NIfTI and DICOM segmentations, DICOM metadata
    bonehub_format.index       build_index
    bonehub_format.publish     write_hub_files
    bonehub_format.webpage     build_webpage
"""

from . import layout
from ._version import SCHEMA_VERSION, __version__, is_compatible_schema_version
from .archive import pack, pack_scan, unpack, unpack_scan
from .edit import update_model_files, update_segmentation
from .labelmap import (
    LabelMap,
    category_of,
    covers,
    is_aggregate,
    kind_of,
    label_for,
    labels_covered,
    members_of,
    node_of,
    parent_of,
    part_of,
    section_of,
    side_of,
    tissue_of,
)
from .licenses import LICENSE_KINDS, LICENSES, RESTRICTED, check_license, is_publishable, is_restricted
from .metadata import (
    LABEL_KINDS,
    DatasetInfo,
    FileInfo,
    LabeledFileInfo,
    Licenses,
    ScanInfo,
    SourceInfo,
    SubjectInfo,
    SubsetInfo,
    check_label,
    check_origin,
    file_license,
    sort_labels,
)
from .reader import Database, DataFile, Dataset, Scan, Subject
from .registry import REGISTRY_URL, load_registry, registry_entry_url
from .segmentation_file import (
    SEGMENTATION_SUFFIX,
    Segment,
    read_label_mask,
    read_segmentation,
    read_segmentation_header,
    read_segmentation_labels,
    read_segmentation_segments,
    segment_number_dtype,
    write_indexed_segmentation,
    write_regions,
    write_segmentation,
)
from .validate import validate

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "is_compatible_schema_version",
    "layout",
    "LabelMap",
    "node_of",
    "part_of",
    "tissue_of",
    "side_of",
    "kind_of",
    "category_of",
    "section_of",
    "parent_of",
    "is_aggregate",
    "members_of",
    "covers",
    "labels_covered",
    "label_for",
    "LICENSES",
    "RESTRICTED",
    "check_license",
    "is_restricted",
    "is_publishable",
    "LICENSE_KINDS",
    "LABEL_KINDS",
    "DatasetInfo",
    "SubjectInfo",
    "ScanInfo",
    "Licenses",
    "SourceInfo",
    "FileInfo",
    "LabeledFileInfo",
    "SubsetInfo",
    "file_license",
    "check_label",
    "check_origin",
    "sort_labels",
    "Database",
    "Dataset",
    "Subject",
    "Scan",
    "DataFile",
    "SEGMENTATION_SUFFIX",
    "Segment",
    "read_label_mask",
    "read_segmentation",
    "read_segmentation_header",
    "read_segmentation_labels",
    "read_segmentation_segments",
    "segment_number_dtype",
    "write_segmentation",
    "write_regions",
    "write_indexed_segmentation",
    "REGISTRY_URL",
    "load_registry",
    "registry_entry_url",
    "update_segmentation",
    "update_model_files",
    "pack",
    "pack_scan",
    "unpack",
    "unpack_scan",
    "validate",
]

__author__ = "Hamid Alavi"
