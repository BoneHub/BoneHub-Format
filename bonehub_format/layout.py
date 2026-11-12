"""Where everything lives in a BoneHub-format database.

    <database>/
    ├── README.md, LICENSE.md, .gitattributes, .gitignore, upload.md     publish.write_hub_files
    ├── index/                                                            index.build_index
    ├── github-webpage/                                                   webpage.build_webpage
    ├── logs/
    └── ds-0001/                                                          named after its BoneHub ID (registry.py)
        ├── dataset.json
        └── sub-000001/
            ├── subject.json
            └── scan-01/
                ├── scan.json
                ├── image.nii.gz
                ├── segmentation.seg.nrrd
                ├── mesh/   <LABEL>.stl                 -> mesh.zip
                └── nurbs/  <LABEL>.iges, <LABEL>.step  -> nurbs.zip

A folder's name is its ID; no file name or metadata file repeats it.
"""

import re
from pathlib import Path

DATASET_JSON = "dataset.json"
SUBJECT_JSON = "subject.json"
SCAN_JSON = "scan.json"
IMAGE_FILE = "image.nii.gz"
SEGMENTATION_FILE = "segmentation.seg.nrrd"

#: Kinds of per-label model files in a scan, with the file types each holds. A scan keeps them in a folder named
#: after the kind (mesh/, nurbs/) and publishes them as a zip of that folder (mesh.zip, nurbs.zip).
MODEL_KINDS: dict[str, tuple[str, ...]] = {
    "mesh": (".stl",),
    "nurbs": (".iges", ".step"),
}

INDEX_DIR = "index"
LOGS_DIR = "logs"
WEBPAGE_DIR = "github-webpage"

#: The largest IDs the folder names hold (ds-NNNN, sub-NNNNNN, scan-NN).
MAX_DATASETS = 9999
MAX_SUBJECTS_PER_DATASET = 999999
MAX_SCANS_PER_SUBJECT = 99

_DATASET_DIR = re.compile(r"ds-(\d{4})")
_SUBJECT_DIR = re.compile(r"sub-(\d{6})")
_SCAN_DIR = re.compile(r"scan-(\d{2})")
_SCAN_KEY = re.compile(r"ds-(\d{4})/sub-(\d{6})/scan-(\d{2})")


def _check_id(kind: str, value: int, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"A {kind} ID must be an integer from 1 to {maximum}, not {value!r}.")
    return value


def dataset_dir_name(dataset_id: int) -> str:
    return f"ds-{_check_id('dataset', dataset_id, MAX_DATASETS):04d}"


def subject_dir_name(subject_id: int) -> str:
    return f"sub-{_check_id('subject', subject_id, MAX_SUBJECTS_PER_DATASET):06d}"


def scan_dir_name(scan_id: int) -> str:
    return f"scan-{_check_id('scan', scan_id, MAX_SCANS_PER_SUBJECT):02d}"


def _parse(pattern: re.Pattern, name: str) -> int | None:
    match = pattern.fullmatch(name)
    return int(match.group(1)) if match and int(match.group(1)) > 0 else None


def parse_dataset_dir(name: str) -> int | None:
    """The dataset ID of a folder name such as 'ds-0027', or None if it is not one."""
    return _parse(_DATASET_DIR, name)


def parse_subject_dir(name: str) -> int | None:
    return _parse(_SUBJECT_DIR, name)


def parse_scan_dir(name: str) -> int | None:
    return _parse(_SCAN_DIR, name)


def dataset_path(root: Path, dataset_id: int) -> Path:
    return Path(root) / dataset_dir_name(dataset_id)


def subject_path(root: Path, dataset_id: int, subject_id: int) -> Path:
    return dataset_path(root, dataset_id) / subject_dir_name(subject_id)


def scan_path(root: Path, dataset_id: int, subject_id: int, scan_id: int) -> Path:
    return subject_path(root, dataset_id, subject_id) / scan_dir_name(scan_id)


def scan_key(dataset_id: int, subject_id: int, scan_id: int) -> str:
    """The path of a scan relative to the database's root folder, e.g. 'ds-0027/sub-000001/scan-01', which names it everywhere."""
    return f"{dataset_dir_name(dataset_id)}/{subject_dir_name(subject_id)}/{scan_dir_name(scan_id)}"


def parse_scan_key(key: str) -> tuple[int, int, int]:
    match = _SCAN_KEY.fullmatch(key)
    if not match:
        raise ValueError(f"'{key}' is not a scan path such as 'ds-0027/sub-000001/scan-01'.")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def model_folder(scan_folder: Path, kind: str) -> Path:
    if kind not in MODEL_KINDS:
        raise ValueError(f"Unknown kind '{kind}'; the kinds are {list(MODEL_KINDS)}.")
    return Path(scan_folder) / kind


def model_zip(scan_folder: Path, kind: str) -> Path:
    return model_folder(scan_folder, kind).with_suffix(".zip")
