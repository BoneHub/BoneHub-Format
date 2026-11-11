"""BoneHub-Format: the BoneHub data format for 3D bone datasets, and the tools to build, check and publish datasets in it.

The base install reads, checks and packs datasets; it needs only pydantic. Converting sources, building the index,
matching scans and writing the Hugging Face files and webpage need the converter extra
(pip install "bonehub-format[converter]") and are imported from their modules:

    bonehub_format.converter   DatasetConverter, SubjectSource, ScanSource
    bonehub_format.export      helpers for converters: images, NIfTI and DICOM segmentations, DICOM metadata
    bonehub_format.index       build_index
    bonehub_format.matching    find_identical_images
    bonehub_format.publish     write_hub_files
    bonehub_format.webpage     build_webpage
"""

from . import layout
from ._version import SCHEMA_VERSION, __version__, is_compatible_schema_version
from .archive import pack, pack_scan, unpack, unpack_scan
from .labelmap import BoneLabelMap, bonehub_to_snomed, part_of, side_of, structure_of, tissue_of
from .licenses import LICENSES, RESTRICTED, check_license, is_restricted
from .links import Link, declared_links, load_links, merge_links, save_links, scan_groups
from .metadata import LABEL_KINDS, DatasetInfo, ScanInfo, SubjectInfo, check_label, check_origin, sort_labels
from .reader import Dataset, Root, Scan, Subject
from .segmentation_file import (
    SEGMENTATION_SUFFIX,
    read_segmentation,
    read_segmentation_header,
    read_segmentation_labels,
    read_segmentation_segments,
    segment_number_dtype,
    write_indexed_segmentation,
    write_segmentation,
)
from .validate import validate

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "is_compatible_schema_version",
    "layout",
    "BoneLabelMap",
    "bonehub_to_snomed",
    "structure_of",
    "part_of",
    "tissue_of",
    "side_of",
    "LICENSES",
    "RESTRICTED",
    "check_license",
    "is_restricted",
    "LABEL_KINDS",
    "DatasetInfo",
    "SubjectInfo",
    "ScanInfo",
    "check_label",
    "check_origin",
    "sort_labels",
    "Root",
    "Dataset",
    "Subject",
    "Scan",
    "SEGMENTATION_SUFFIX",
    "read_segmentation",
    "read_segmentation_header",
    "read_segmentation_labels",
    "read_segmentation_segments",
    "segment_number_dtype",
    "write_segmentation",
    "write_indexed_segmentation",
    "Link",
    "load_links",
    "save_links",
    "merge_links",
    "declared_links",
    "scan_groups",
    "pack",
    "pack_scan",
    "unpack",
    "unpack_scan",
    "validate",
]

__author__ = "Hamid Alavi"
