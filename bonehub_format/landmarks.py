"""Landmarks: points, axes, spheres, frames and areas on the structures of a scan, kept in landmarks-NN.zip.

    landmarks-01/                   published as landmarks-01.zip, described by landmarks-01.json
        landmarks.json              every landmark: its name, type and numbers
        areas.nrrd                  (optional) areas as masks on the scan's image grid, a segmentation file
        <LANDMARK>[-<k>].stl        (optional) areas as surface patches

landmarks.json holds coordinates in LPS millimetres, in the frame of the scan's image (like the meshes):

    {"coordinate_system": "LPS", "units": "mm", "landmarks": [
        {"name": "LANDMARK_FEMUR_HEAD_CENTER_LEFT", "type": "sphere", "center": [-84.2, -131.7, -318.9], "radius": 22.7},
        {"name": "LANDMARK_HIP_BONE_ANTERIOR_SUPERIOR_ILIAC_SPINE_RIGHT", "type": "point",
         "position": [112.4, -88.1, -221.6], "rater": 3, "session": 2, "source_name": "Right ASIS"},
        {"name": "LANDMARK_GLUTEUS_MEDIUS_ORIGIN_LEFT", "type": "area", "file": "areas.nrrd"}]}

An entry's name is a landmark label (LANDMARK_..., see labelmap.py), its type one of the geometries that label allows
(labelmap.geometry_of), and its numbers those of its type:

    point     position [x, y, z]
    points    positions [[x, y, z], ...]           several points, in order (a digitised outline, a muscle's path)
    line      points [[x, y, z], [x, y, z]]        an axis through two points
    cylinder  points (two points of its axis), radius
    circle    center, radius, normal ([x, y, z], or null when the source gives none)
    sphere    center, radius
    frame     origin, axes [[...], [...], [...]]   three orthonormal unit vectors, the x, y and z axes
    area      file: "areas.nrrd" (the segment of the entry's name) or the name of an STL file of the folder

Optional fields: rater and session (integers, for repeated placements of one landmark), element (a source's own
subdivision, e.g. the number of a muscle's fascicle), source_name (the source's name for it), values (other numbers
the source gives with it, e.g. principal moments of inertia, as a name -> number object) and remarks.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from pathlib import Path

from .labelmap import TABLES as _TABLES
from .metadata import sort_labels, write_json

LANDMARKS_JSON = "landmarks.json"
AREAS_FILE = "areas.nrrd"
COORDINATE_SYSTEM, UNITS = "LPS", "mm"

#: The numbers of each type: field -> what it holds ('xyz' one point, 'xyz*' several, 'xyz2' two, 'xyz3' three unit
#: vectors, 'r' a positive number, 'xyz?' one point or null, 'file' a file of the folder).
TYPES = {
    "point": {"position": "xyz"},
    "points": {"positions": "xyz*"},
    "line": {"points": "xyz2"},
    "cylinder": {"points": "xyz2", "radius": "r"},
    "circle": {"center": "xyz", "radius": "r", "normal": "xyz?"},
    "sphere": {"center": "xyz", "radius": "r"},
    "frame": {"origin": "xyz", "axes": "xyz3"},
    "area": {"file": "file"},
}
OPTIONAL = ("rater", "session", "element", "source_name", "values", "remarks")
#: An area's STL file: the landmark's name, and a number where it has several patches.
_STL_NAME = re.compile(r"(LANDMARK_[A-Z0-9_]+?)(?:-(\d+))?\.stl")


def _xyz(value) -> bool:
    return (isinstance(value, list) and len(value) == 3
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value))


def _check_value(kind: str, value, files: set[str] | None) -> str | None:
    """What is wrong with a value of a kind (see TYPES), or None."""
    if kind == "xyz":
        return None if _xyz(value) else "is not [x, y, z]"
    if kind == "xyz?":
        return None if value is None or _xyz(value) else "is not [x, y, z] or null"
    if kind == "xyz*":
        return None if isinstance(value, list) and value and all(_xyz(v) for v in value) else "is not a list of [x, y, z]"
    if kind == "xyz2":
        return None if isinstance(value, list) and len(value) == 2 and all(_xyz(v) for v in value) else "is not two [x, y, z]"
    if kind == "xyz3":
        if not (isinstance(value, list) and len(value) == 3 and all(_xyz(v) for v in value)):
            return "is not three [x, y, z]"
        for i in range(3):
            for j in range(3):
                dot = sum(a * b for a, b in zip(value[i], value[j]))
                if abs(dot - (1.0 if i == j else 0.0)) > 1e-3:
                    return "is not three orthonormal unit vectors"
        return None
    if kind == "r":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
        return None if ok else "is not a positive number"
    if kind == "file":
        if not isinstance(value, str) or not (value == AREAS_FILE or _STL_NAME.fullmatch(value)):
            return f"is not {AREAS_FILE} or <LANDMARK>[-k].stl"
        if files is not None and value not in files:
            return "names a file the folder does not hold"
        return None
    raise ValueError(kind)


def entry_problems(entry, files: set[str] | None = None) -> list[str]:
    """What is wrong with one entry of landmarks.json; files are the names of the files next to it, if known."""
    if not isinstance(entry, dict):
        return ["an entry is not an object"]
    name, kind = entry.get("name"), entry.get("type")
    where = f"landmark {name!r}"
    if not isinstance(name, str) or name not in _TABLES.labels or not _TABLES.is_landmark(name):
        return [f"{where}: not a landmark label (LANDMARK_..., see labelmap.py)"]
    allowed = _TABLES.geometry_of(name)
    if kind not in TYPES:
        return [f"{where}: type {kind!r} is not one of {list(TYPES)}"]
    p = []
    if kind not in allowed:
        p.append(f"{where}: is a {kind}, but {name} is one of {list(allowed)}")
    fields = TYPES[kind]
    for field, value_kind in fields.items():
        if field not in entry:
            p.append(f"{where}: a {kind} needs '{field}'")
            continue
        wrong = _check_value(value_kind, entry[field], files)
        if wrong:
            p.append(f"{where}: '{field}' {wrong}")
    if kind == "area" and isinstance(entry.get("file"), str) and entry["file"] != AREAS_FILE:
        match = _STL_NAME.fullmatch(entry["file"])
        if match and match.group(1) != name:
            p.append(f"{where}: its file {entry['file']} is named for another landmark")
    extra = set(entry) - {"name", "type", *fields, *OPTIONAL}
    if extra:
        p.append(f"{where}: unknown fields {sorted(extra)}")
    for field in ("rater", "session"):
        if field in entry and (type(entry[field]) is not int or entry[field] < 1):
            p.append(f"{where}: '{field}' is not a positive integer")
    if "element" in entry and not (isinstance(entry["element"], str) or type(entry["element"]) is int):
        p.append(f"{where}: 'element' is not a text or an integer")
    for field in ("source_name", "remarks"):
        if field in entry and not isinstance(entry[field], str):
            p.append(f"{where}: '{field}' is not a text")
    if "values" in entry:
        values = entry["values"]
        if not (isinstance(values, dict) and all(isinstance(k, str) and isinstance(v, (int, float, list, str))
                                                 for k, v in values.items())):
            p.append(f"{where}: 'values' is not an object of names and numbers")
    return p


def problems(document, files: set[str] | None = None) -> list[str]:
    """What is wrong with the content of a landmarks.json; files are the names of the files next to it, if known."""
    if not isinstance(document, dict):
        return ["landmarks.json does not hold an object"]
    p = []
    if document.get("coordinate_system") != COORDINATE_SYSTEM or document.get("units") != UNITS:
        p.append(f"landmarks.json: coordinates must be {COORDINATE_SYSTEM} in {UNITS}")
    entries = document.get("landmarks")
    if not isinstance(entries, list) or not entries:
        return p + ["landmarks.json: 'landmarks' is not a list of entries"]
    extra = set(document) - {"coordinate_system", "units", "landmarks"}
    if extra:
        p.append(f"landmarks.json: unknown fields {sorted(extra)}")
    for entry in entries:
        p.extend(entry_problems(entry, files))
    return p


def labels(document: dict) -> list[str]:
    """The landmarks a document holds, in label map order."""
    return sort_labels({entry["name"] for entry in document["landmarks"]})


def write_landmarks(folder: Path, entries: list[dict]) -> Path:
    """Write landmarks.json into a landmarks folder (made if needed), after checking the entries against the files the
    folder holds (write its area files first). Return the file written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    document = {"coordinate_system": COORDINATE_SYSTEM, "units": UNITS, "landmarks": entries}
    found = problems(document, {f.name for f in folder.iterdir() if f.is_file()})
    if found:
        raise ValueError("The landmarks are not valid:\n" + "\n".join(found))
    path = folder / LANDMARKS_JSON
    write_json(path, document)
    return path


def read_landmarks(source: Path | zipfile.Path) -> dict:
    """The content of landmarks.json, from a landmarks folder, a landmarks zip, or the file itself."""
    if isinstance(source, zipfile.Path):
        return json.loads(source.read_text(encoding="utf-8"))
    source = Path(source)
    if source.is_dir():
        source = source / LANDMARKS_JSON
    elif source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            return json.loads(archive.read(LANDMARKS_JSON).decode("utf-8"))
    return json.loads(source.read_text(encoding="utf-8"))


def is_landmarks_file(name: str) -> bool:
    """Whether a file name belongs in a landmarks folder: landmarks.json, areas.nrrd or <LANDMARK>[-k].stl."""
    if name in (LANDMARKS_JSON, AREAS_FILE):
        return True
    match = _STL_NAME.fullmatch(name)
    return bool(match) and match.group(1) in _TABLES.labels and _TABLES.is_landmark(match.group(1))
