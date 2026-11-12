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
from .labelmap import BoneLabelMap, bonehub_to_snomed, part_of, side_of, structure_of, tissue_of
from .licenses import LICENSES, RESTRICTED, check_license, is_restricted
from .metadata import LABEL_KINDS, DatasetInfo, ScanInfo, SubjectInfo, check_label, check_origin, sort_labels
from .reader import Database, Dataset, Scan, Subject
from .registry import REGISTRY_URL, load_registry
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
    "Database",
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
    "REGISTRY_URL",
    "load_registry",
    "update_segmentation",
    "update_model_files",
    "pack",
    "pack_scan",
    "unpack",
    "unpack_scan",
    "validate",
]

__author__ = "Hamid Alavi"
