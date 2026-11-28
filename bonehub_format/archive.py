"""The zips in which a scan's per-label model files and its landmarks are published: mesh-01.zip, nurbs-01.zip,
landmarks-01.zip, ...

A scan keeps them in numbered folders of their kind (mesh-01/, nurbs-01/, landmarks-01/): one file per label and file
type for the models, landmarks.json and its area files for the landmarks (see landmarks.py). pack() writes each folder into a zip of the same name; the zips are published and the folders are not. unpack() does the reverse for a downloaded copy. A zip is rewritten only when its
folder has changed, and the same files always give the same zip.
"""

import os
import tempfile
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import layout
from .landmarks import AREAS_FILE, LANDMARKS_JSON, is_landmarks_file
from .metadata import check_label

#: Files the operating system leaves in folders; they are neither packed nor reported.
_IGNORED = {"desktop.ini", "thumbs.db", ".ds_store"}
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)
#: Subjects packed or unpacked at once: the time is spent reading and writing files, often on a network share.
THREADS = 8


def belongs_in(kind: str, name: str) -> bool:
    """Whether a file name belongs in a folder of a kind: <LABEL><suffix> of a model kind, or a landmarks file."""
    if kind == "landmarks":
        return is_landmarks_file(name)
    path = Path(name)
    if path.suffix.lower() not in layout.MODEL_KINDS[kind]:
        return False
    try:
        check_label(path.stem)
    except ValueError:
        return False
    return True


def model_folder_files(folder: Path, kind: str) -> list[Path]:
    """The files of a folder of a folder kind, checked (see belongs_in)."""
    files = []
    for entry in sorted(Path(folder).iterdir()):
        if entry.name.lower() in _IGNORED:
            continue
        if not entry.is_file() or not belongs_in(kind, entry.name):
            what = (f"{LANDMARKS_JSON}, {AREAS_FILE} and <LANDMARK>.stl files" if kind == "landmarks"
                    else f"<LABEL>{'/'.join(layout.MODEL_KINDS[kind])} files")
            raise ValueError(f"'{entry}' does not belong in a {kind} folder, which holds only {what}.")
        files.append(entry)
    if kind == "landmarks" and files and not any(f.name == LANDMARKS_JSON for f in files):
        raise ValueError(f"The landmarks folder '{folder}' has no {LANDMARKS_JSON}.")
    return files


def _crc32(path: Path) -> int:
    crc = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            crc = zlib.crc32(chunk, crc)
    return crc


def zip_is_current(folder: Path, zip_path: Path, kind: str) -> bool:
    """True if the zip holds exactly the files of the folder, with the same contents."""
    if not zip_path.is_file():
        return False
    expected = {f.name: (f.stat().st_size, _crc32(f)) for f in model_folder_files(folder, kind)}
    with zipfile.ZipFile(zip_path) as archive:
        found = {info.filename: (info.file_size, info.CRC) for info in archive.infolist()}
    return found == expected


def model_entries(scan_folder: Path) -> list[tuple[str, int, Path, Path]]:
    """(kind, number, folder, zip) for every model folder or zip of a scan, in order."""
    found = {}
    for entry in Path(scan_folder).iterdir():
        parsed = layout.parse_scan_entry(entry.name)
        if parsed is None or parsed[0] not in layout.FOLDER_KINDS or parsed[2] == "json":
            continue
        kind, number, _ = parsed
        found[(kind, number)] = (layout.model_folder(scan_folder, kind, number), layout.model_zip(scan_folder, kind, number))
    return [(kind, number, *paths) for (kind, number), paths in sorted(found.items())]


def pack_folder(folder: Path, kind: str) -> bool:
    """Write <folder>.zip from a folder (e.g. mesh-01/) unless it is current; return True if it was written.

    An empty or missing folder leaves no zip.
    """
    folder = Path(folder)
    zip_path = folder.parent / f"{folder.name}.zip"
    files = model_folder_files(folder, kind) if folder.is_dir() else []
    if not files:
        if folder.is_dir():
            zip_path.unlink(missing_ok=True)
        return False
    if zip_is_current(folder, zip_path, kind):
        return False
    fd, temp_path = tempfile.mkstemp(dir=folder.parent, prefix=f".{zip_path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        with zipfile.ZipFile(temp_path, "w") as archive:
            for path in files:
                info = zipfile.ZipInfo(path.name, date_time=_FIXED_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 0
                info.external_attr = 0
                with open(path, "rb") as source, archive.open(info, "w") as target:
                    while chunk := source.read(1 << 20):
                        target.write(chunk)
        os.replace(temp_path, zip_path)
    except BaseException:
        Path(temp_path).unlink(missing_ok=True)
        raise
    return True


def pack_scan(scan_folder: Path) -> list[Path]:
    """Write the zips of a scan's folders that are not current; return the zips written."""
    written = []
    for kind, _, folder, zip_path in model_entries(scan_folder):
        if folder.is_dir() and pack_folder(folder, kind):
            written.append(zip_path)
    return written


def unpack_scan(scan_folder: Path, overwrite: bool = False) -> list[Path]:
    """Extract the scan's zips into folders that do not exist yet (or into all, with overwrite); return the folders
    written."""
    written = []
    for kind, _, folder, zip_path in model_entries(scan_folder):
        if not zip_path.is_file() or (folder.is_dir() and not overwrite):
            continue
        folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                name = Path(info.filename)
                if name.name != info.filename or not belongs_in(kind, name.name):
                    raise ValueError(f"'{zip_path}' holds '{info.filename}', which does not belong in it.")
                with archive.open(info) as source, open(folder / name.name, "wb") as target:
                    while chunk := source.read(1 << 20):
                        target.write(chunk)
        written.append(folder)
    return written


def _each_subject(root: Path, work) -> list[Path]:
    """Apply work(scan folder) -> paths to every scan of a database, a subject per thread (its folders are listed in
    the thread too); the paths in the order of the scans."""
    from .reader import Database, Subject

    def subject(dataset, subject_id: int) -> list[Path]:
        try:
            scans = Subject(dataset, subject_id).scans()
        except FileNotFoundError:
            return []  # not a subject folder
        return [path for scan in scans for path in work(scan.path)]

    jobs = [
        (dataset, subject_id)
        for dataset in Database(root).datasets()
        for subject_id in sorted(i for i in (layout.parse_subject_dir(e.name) for e in dataset.path.iterdir()) if i is not None)
    ]
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        return [path for paths in pool.map(lambda job: subject(*job), jobs) for path in paths]


def pack(root: Path) -> list[Path]:
    """Bring every zip of a database up to date with its folder; return the zips written."""
    return _each_subject(root, pack_scan)


def unpack(root: Path, overwrite: bool = False) -> list[Path]:
    """Extract every zip of a database into its folder (see unpack_scan); return the folders written."""
    return _each_subject(root, lambda folder: unpack_scan(folder, overwrite))
