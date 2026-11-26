"""Where everything lives in a BoneHub-format database.

    <database>/
    ├── README.md, LICENSE.md, .gitattributes, .gitignore, upload.md     publish.write_hub_files
    ├── index/                                                            index.build_index
    ├── github-webpage/                                                   webpage.build_webpage
    ├── logs/
    └── ds-0001/                          a dataset, numbered within the database
        ├── dataset.json                  the dataset and the registry datasets it is made from (its sources)
        └── sub-000001/
            ├── subject.json
            └── scan-01/
                ├── scan.json             the scan: modality, age, weight, ...
                ├── image.nii.gz          + image.json
                ├── segmentation-1.nrrd   + segmentation-1.json
                ├── segmentation-2.nrrd   + segmentation-2.json
                ├── mesh-1/   <LABEL>.stl                 -> mesh-1.zip   + mesh-1.json
                └── nurbs-1/  <LABEL>.iges, <LABEL>.step  -> nurbs-1.zip  + nurbs-1.json

A folder's name is its ID. A dataset's number has no meaning outside its database: the registry datasets it was made
from are listed in its dataset.json. Every data file has a JSON file of the same name that says which source it came
from, under which license, and (for segmentations, meshes and NURBS) which labels it holds. The files of one kind are
numbered within the scan (segmentation-1, segmentation-2, ...), whatever their source.
"""

import re
from pathlib import Path

DATASET_JSON = "dataset.json"
SUBJECT_JSON = "subject.json"
SCAN_JSON = "scan.json"
IMAGE_FILE = "image.nii.gz"
IMAGE_JSON = "image.json"
SEGMENTATION_SUFFIX = ".nrrd"

#: Kinds of per-label model files in a scan, with the file types each holds. A scan keeps them in a numbered folder
#: (mesh-1/) and publishes them as a zip of that folder (mesh-1.zip).
MODEL_KINDS: dict[str, tuple[str, ...]] = {
    "mesh": (".stl",),
    "nurbs": (".iges", ".step"),
}

#: Kinds of numbered data files, which hold labels.
LABELED_KINDS: tuple[str, ...] = ("segmentation", *MODEL_KINDS)

#: Every kind of data file of a scan.
DATA_KINDS: tuple[str, ...] = ("image", *LABELED_KINDS)

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
_NUMBERED = re.compile(r"(" + "|".join(LABELED_KINDS) + r")-([1-9]\d*)(\.nrrd|\.zip|\.json)?")


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
    """The dataset ID of a folder name such as 'ds-0001', or None if it is not one."""
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
    """The path of a scan relative to the database's root folder, e.g. 'ds-0001/sub-000001/scan-01', which names it everywhere."""
    return f"{dataset_dir_name(dataset_id)}/{subject_dir_name(subject_id)}/{scan_dir_name(scan_id)}"


def parse_scan_key(key: str) -> tuple[int, int, int]:
    match = _SCAN_KEY.fullmatch(key)
    if not match:
        raise ValueError(f"'{key}' is not a scan path such as 'ds-0001/sub-000001/scan-01'.")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _check_kind(kind: str, kinds=DATA_KINDS) -> str:
    if kind not in kinds:
        raise ValueError(f"Unknown kind '{kind}'; the kinds are {list(kinds)}.")
    return kind


def _check_number(kind: str, number: int | None) -> int | None:
    if kind == "image":
        if number is not None:
            raise ValueError("A scan has one image, which is not numbered.")
        return None
    if type(number) is not int or number < 1:
        raise ValueError(f"The {kind} files of a scan are numbered from 1, not {number!r}.")
    return number


def data_file(scan_folder: Path, kind: str, number: int | None = None) -> Path:
    """The data file of a kind that is published: image.nii.gz, segmentation-2.nrrd, mesh-1.zip, nurbs-1.zip."""
    number = _check_number(_check_kind(kind), number)
    if kind == "image":
        return Path(scan_folder) / IMAGE_FILE
    if kind == "segmentation":
        return Path(scan_folder) / f"segmentation-{number}{SEGMENTATION_SUFFIX}"
    return Path(scan_folder) / f"{kind}-{number}.zip"


def data_json(scan_folder: Path, kind: str, number: int | None = None) -> Path:
    """The JSON file that describes a data file: image.json, segmentation-2.json, mesh-1.json."""
    number = _check_number(_check_kind(kind), number)
    return Path(scan_folder) / (IMAGE_JSON if kind == "image" else f"{kind}-{number}.json")


def model_folder(scan_folder: Path, kind: str, number: int) -> Path:
    """The folder of a scan's model files of a kind: mesh-1/ (published as mesh-1.zip)."""
    return Path(scan_folder) / f"{_check_kind(kind, MODEL_KINDS)}-{_check_number(kind, number)}"


def model_zip(scan_folder: Path, kind: str, number: int) -> Path:
    """The zip of a model folder: mesh-1.zip."""
    return data_file(scan_folder, _check_kind(kind, MODEL_KINDS), number)


def parse_scan_entry(name: str) -> tuple[str, int | None, str] | None:
    """What a file or folder of a scan folder is, from its name: (kind, number, role), the role being 'data' (the
    published file: image.nii.gz, segmentation-1.nrrd, mesh-1.zip), 'json' (its JSON file) or 'folder' (a model
    folder, mesh-1/). The image has no number. None for any other name, scan.json included."""
    if name == IMAGE_FILE:
        return "image", None, "data"
    if name == IMAGE_JSON:
        return "image", None, "json"
    match = _NUMBERED.fullmatch(name)
    if match is None:
        return None
    kind, number, suffix = match.group(1), int(match.group(2)), match.group(3)
    if suffix == ".json":
        return kind, number, "json"
    if kind == "segmentation":
        return (kind, number, "data") if suffix == SEGMENTATION_SUFFIX else None
    if suffix == ".zip":
        return kind, number, "data"
    return (kind, number, "folder") if suffix is None else None
