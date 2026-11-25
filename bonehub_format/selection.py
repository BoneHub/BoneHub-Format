"""One scan per subject: choosing the scan to keep when the source has several of the same subject.

A BoneHub database aims at many different subjects, not at many scans of the same person: of the contrast phases,
reconstructions, follow-ups or test/retest scans a source has of one subject, one is converted. The scan kept is the
one with the most labelled data (as the converter counts it, e.g. the number of labelled structures, or of label sets
from other datasets), then the one that covers most of the body (its extent along the body axis), then the one with
the thinnest slices, then the earliest. A second scan is kept only where it holds other data that cannot be put on
the first scan's grid (e.g. the meshes of another dataset, made in another frame).
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from bonehub_format.converter import ScanSource


def _dicom_geometry(folder: Path) -> tuple[float, float] | None:
    """Body coverage and slice spacing of a DICOM series, from the positions of its first and last slice."""
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder))
    if len(files) < 2:
        return None
    positions = []
    for name in (files[0], files[-1]):
        reader = sitk.ImageFileReader()
        reader.SetFileName(name)
        reader.ReadImageInformation()
        positions.append(np.array(reader.GetOrigin()))
    coverage = float(np.abs(positions[1] - positions[0])[2])
    spacing = coverage / (len(files) - 1)
    return coverage + spacing, spacing


def image_geometry(path: Path) -> tuple[float, float] | None:
    """The extent of an image (a file or a DICOM folder) along the body (head-feet) axis in mm and its slice
    thickness along that axis, from the header; None if it can't be read."""
    path = Path(path)
    if path.is_dir():
        try:
            return _dicom_geometry(path)
        except RuntimeError:
            return None
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    try:
        reader.ReadImageInformation()
    except RuntimeError:
        return None
    direction = np.reshape(reader.GetDirection(), (3, 3))
    size, spacing = np.array(reader.GetSize()), np.array(reader.GetSpacing())
    axis = int(np.argmax(np.abs(direction[2])))  # the image axis closest to the body axis
    return float(np.sum(np.abs(direction[2]) * size * spacing)), float(spacing[axis])


def _header_geometry(scan: ScanSource) -> tuple[float, float] | None:
    return image_geometry(scan.image_path) if scan.image_path is not None else None


Data = Callable[[ScanSource], float]
Geometry = Callable[[ScanSource], tuple[float, float] | None]


def choose_scan(scans: list[ScanSource], data: Data = lambda scan: 0, geometry: Geometry = _header_geometry) -> ScanSource:
    """The scan to keep of several scans of one subject: most data, then most body coverage, then thinnest slices,
    then the earliest (days_from_baseline, else the order given). geometry(scan) gives (coverage, slice thickness) in
    mm; by default from the image header."""
    if len(scans) == 1:
        return scans[0]

    def rank(item):
        order, scan = item
        found = geometry(scan)
        coverage, thickness = found if found else (0.0, float("inf"))
        days = scan.info.days_from_baseline
        return (-data(scan), -round(coverage), round(thickness, 3), days if days is not None else 0, order)

    return min(enumerate(scans), key=rank)[1]


def left_out_remark(kept: ScanSource, scans: list[ScanSource], what: str = "scans") -> str | None:
    """The remark for a kept scan that says how many others of the subject were left out."""
    others = len(scans) - 1
    if others == 0:
        return None
    return f"The source has {others} other {what if others > 1 else what.rstrip('s')} of this subject, not converted (one scan per subject)."


def keep_one(scans: list[ScanSource], data: Data = lambda scan: 0, geometry: Geometry = _header_geometry, what: str = "scans") -> list[ScanSource]:
    """[the chosen scan], with the remark on the others left out added to its remarks."""
    kept = choose_scan(scans, data, geometry)
    remark = left_out_remark(kept, scans, what)
    if remark:
        kept.info.remarks = f"{kept.info.remarks} {remark}" if kept.info.remarks else remark
    return [kept]
