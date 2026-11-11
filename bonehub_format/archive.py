"""The zips in which a scan's per-label model files are published: mesh.zip and nurbs.zip.

A scan keeps its model files in folders (mesh/, nurbs/), one file per label and file type. pack() writes each folder
into a zip of the same name; the zips are published and the folders are not. unpack() does the reverse for a
downloaded copy. A zip is rewritten only when its folder has changed, and the same files always give the same zip.
"""

import os
import tempfile
import zipfile
import zlib
from pathlib import Path

from . import layout
from .metadata import check_label

#: Files the operating system leaves in folders; they are neither packed nor reported.
_IGNORED = {"desktop.ini", "thumbs.db", ".ds_store"}
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def model_folder_files(folder: Path, kind: str) -> list[Path]:
    """The files of a model folder, checked: each is named <LABEL><suffix> with a suffix of its kind."""
    suffixes = layout.MODEL_KINDS[kind]
    files = []
    for entry in sorted(Path(folder).iterdir()):
        if entry.name.lower() in _IGNORED:
            continue
        if not entry.is_file() or entry.suffix.lower() not in suffixes:
            raise ValueError(f"'{entry}' does not belong in a {kind} folder, which holds only {', '.join(suffixes)} files.")
        check_label(entry.stem)
        files.append(entry)
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


def pack_folder(folder: Path, kind: str) -> bool:
    """Write <kind>.zip from the scan's <kind>/ folder unless it is current; return True if it was written.

    An empty or missing folder leaves no zip.
    """
    folder = Path(folder)
    zip_path = folder.with_suffix(".zip")
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
    """Write the zips of a scan's model folders that are not current; return the zips written."""
    written = []
    for kind in layout.MODEL_KINDS:
        folder = layout.model_folder(scan_folder, kind)
        if pack_folder(folder, kind):
            written.append(folder.with_suffix(".zip"))
    return written


def unpack_scan(scan_folder: Path, overwrite: bool = False) -> list[Path]:
    """Extract the scan's model zips into folders that do not exist yet (or into all, with overwrite); return the
    folders written."""
    written = []
    for kind in layout.MODEL_KINDS:
        folder = layout.model_folder(scan_folder, kind)
        zip_path = layout.model_zip(scan_folder, kind)
        if not zip_path.is_file() or (folder.is_dir() and not overwrite):
            continue
        folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                name = Path(info.filename)
                if name.name != info.filename or name.suffix.lower() not in layout.MODEL_KINDS[kind]:
                    raise ValueError(f"'{zip_path}' holds '{info.filename}', which does not belong in it.")
                with archive.open(info) as source, open(folder / name.name, "wb") as target:
                    while chunk := source.read(1 << 20):
                        target.write(chunk)
        written.append(folder)
    return written


def _scan_folders(root: Path):
    from .reader import Root

    for scan in Root(root).scans():
        yield scan.path


def pack(root: Path) -> list[Path]:
    """Bring every zip of a root up to date with its folder; return the zips written."""
    written = []
    for scan_folder in _scan_folders(root):
        written += pack_scan(scan_folder)
    return written


def unpack(root: Path, overwrite: bool = False) -> list[Path]:
    """Extract every zip of a root into its folder (see unpack_scan); return the folders written."""
    written = []
    for scan_folder in _scan_folders(root):
        written += unpack_scan(scan_folder, overwrite)
    return written
