"""Changing a dataset in place: adding, replacing and removing the labels of a scan.

Whether a change is made in place or published as a new dataset is up to whoever maintains the dataset (see the
README). In place, the dataset keeps its ID, and the origin each label has in scan.json tells the labels of the source
from those added later: a bone added by a model is model:<name>@<version>, one drawn or corrected by hand is manual.

These functions keep scan.json in step with the files. They leave dataset.json alone: when a label is now segmented
in every scan of the dataset, add it to annotated_labels yourself. Rebuild the index afterwards.

    update_segmentation(scan, mask, origin="model:TotalSegmentator@2.4.0")   # add or replace the labels of mask
    update_segmentation(scan, remove=["PATELLA_LEFT"])
    update_model_files(scan, "mesh", {"FEMUR_LEFT": Path("femur.stl")}, origin="derived:my-mesher@1.0")
    update_model_files(scan, "mesh", remove=["FEMUR_LEFT"])

`scan` is a Scan (see reader.py) or the path of a scan folder. update_segmentation needs numpy and SimpleITK:
pip install "bonehub-format[io]".
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from . import layout
from .archive import model_folder_files, pack_scan, unpack_scan
from .labelmap import BoneLabelMap
from .metadata import ScanInfo, check_label, check_origin
from .reader import Scan
from .segmentation_file import SEGMENTATION_SUFFIX, _io_modules, read_segmentation, write_segmentation

if TYPE_CHECKING:
    import SimpleITK as sitk

Origin = str | Mapping[str, str]


def _scan_folder(scan: Scan | Path) -> Path:
    folder = scan.path if isinstance(scan, Scan) else Path(scan)
    if not (folder / layout.SCAN_JSON).is_file():
        raise FileNotFoundError(f"'{folder}' has no {layout.SCAN_JSON}.")
    return folder


def _new_origins(labels: set[str], origin: Origin | None, kind: str) -> dict[str, str]:
    """The origin of each label being added, checked before any file is changed."""
    if not labels:
        return {}
    if origin is None:
        raise ValueError(f"Give the origin of the {kind} being added (see check_origin).")
    if isinstance(origin, str):
        origins = {label: origin for label in labels}
    else:
        missing = sorted(labels - set(origin))
        if missing:
            raise ValueError(f"No origin given for the {kind} of {missing}.")
        origins = {label: origin[label] for label in labels}
    for value in origins.values():
        check_origin(value)
    return origins


def _record(scan: Scan | Path, folder: Path, info: ScanInfo, kind: str, present: Iterable[str], added: dict[str, str]):
    """Write the labels of a kind into scan.json: the added ones with their new origin, the others as they were."""
    before = getattr(info, kind) or {}
    origins = {}
    for label in present:
        if label in added:
            origins[label] = added[label]
        elif label in before:
            origins[label] = before[label]
        else:
            raise ValueError(f"'{folder}' has {kind} data for {label}, which scan.json does not list; fix the scan first.")
    info.set_labels(kind, list(origins), origins)
    info.save(folder / layout.SCAN_JSON)
    if isinstance(scan, Scan):
        scan.__dict__.pop("info", None)  # read again when next used


def _grid(folder: Path, mask: sitk.Image | None) -> sitk.Image | None:
    """An empty image with the geometry of the scan's image (or of the mask, for a scan without one)."""
    _, sitk = _io_modules()
    image_path = folder / layout.IMAGE_FILE
    if image_path.is_file():
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(image_path))
        reader.ReadImageInformation()
        size, spacing, origin, direction = reader.GetSize(), reader.GetSpacing(), reader.GetOrigin(), reader.GetDirection()
    elif mask is not None:
        size, spacing, origin, direction = mask.GetSize(), mask.GetSpacing(), mask.GetOrigin(), mask.GetDirection()
    else:
        return None
    grid = sitk.Image(list(size), sitk.sitkInt32)
    grid.SetSpacing(spacing)
    grid.SetOrigin(origin)
    grid.SetDirection(direction)
    return grid


def _check_same_grid(mask: sitk.Image, grid: sitk.Image) -> None:
    np, _ = _io_modules()
    same = (
        mask.GetSize() == grid.GetSize()
        and np.allclose(mask.GetSpacing(), grid.GetSpacing(), rtol=1e-4, atol=1e-4)
        and np.allclose(mask.GetOrigin(), grid.GetOrigin(), rtol=0, atol=1e-3)
        and np.allclose(mask.GetDirection(), grid.GetDirection(), rtol=0, atol=1e-4)
    )
    if not same:
        raise ValueError(
            "The mask is not on the grid of the scan's image: size, spacing, origin and direction are "
            f"{mask.GetSize()}, {mask.GetSpacing()}, {mask.GetOrigin()}, {mask.GetDirection()} instead of "
            f"{grid.GetSize()}, {grid.GetSpacing()}, {grid.GetOrigin()}, {grid.GetDirection()}."
        )


def update_segmentation(
    scan: Scan | Path,
    mask: sitk.Image | Path | None = None,
    origin: Origin | None = None,
    remove: Iterable[str] = (),
    overlap: Literal["replace", "keep"] = "replace",
) -> list[str]:
    """Add or replace the labels of `mask` in the scan's segmentation, and take out the labels in `remove`.

    mask: BoneLabelMap values (0 for background) on the grid of the scan's image, as a SimpleITK image or the path of
        a file SimpleITK reads (a .seg.nrrd is read with read_segmentation). Each label it holds replaces that label.
    origin: where the labels of the mask come from (see check_origin), one for all or one per label.
    remove: labels to take out.
    overlap: where a label of the mask covers a voxel of another label, 'replace' gives the voxel to the mask's label
        and 'keep' leaves it to the label already there.
    Returns the labels of the segmentation afterwards. A label left without voxels is dropped from scan.json, and a
    segmentation left without labels is deleted.
    """
    np, sitk = _io_modules()
    folder = _scan_folder(scan)
    info = ScanInfo.load(folder / layout.SCAN_JSON)
    remove = {check_label(label) for label in remove}
    if overlap not in ("replace", "keep"):
        raise ValueError(f"overlap is 'replace' or 'keep', not {overlap!r}.")

    if mask is not None and not hasattr(mask, "GetSize"):
        path = Path(mask)
        mask = read_segmentation(path) if path.name.endswith(SEGMENTATION_SUFFIX) else sitk.ReadImage(str(path))
    segmentation_path = folder / layout.SEGMENTATION_FILE
    if segmentation_path.is_file():
        current_image = read_segmentation(segmentation_path)
    else:
        current_image = _grid(folder, mask)
        if current_image is None:
            return []  # nothing to remove from, nothing to add
    current = sitk.GetArrayFromImage(current_image).astype(np.int64)

    new = None
    added: dict[str, str] = {}
    if mask is not None:
        _check_same_grid(mask, current_image)
        new = sitk.GetArrayFromImage(mask).astype(np.int64)
        values = [int(v) for v in np.unique(new) if v != 0]
        try:
            new_labels = {BoneLabelMap(value).name for value in values}
        except ValueError as e:
            raise ValueError(f"The mask holds a value that is not a BoneLabelMap label: {e}") from e
        both = sorted(new_labels & remove)
        if both:
            raise ValueError(f"{both} are both added and removed.")
        added = _new_origins(new_labels, origin, "segmentation")

    for label in remove | set(added):
        current[current == BoneLabelMap[label].value] = 0
    if new is not None:
        take = new != 0 if overlap == "replace" else (new != 0) & (current == 0)
        current[take] = new[take]

    if current.any():
        temporary = folder / f".{os.getpid()}.tmp{SEGMENTATION_SUFFIX}"
        try:
            present = write_segmentation(current, current_image, temporary)
            os.replace(temporary, segmentation_path)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        segmentation_path.unlink(missing_ok=True)
        present = []
    _record(scan, folder, info, "segmentation", present, added)
    return info.labels("segmentation")


def update_model_files(
    scan: Scan | Path,
    kind: str,
    files: Mapping[str, Path | Sequence[Path]] | None = None,
    origin: Origin | None = None,
    remove: Iterable[str] = (),
) -> list[str]:
    """Add or replace the files of labels in the scan's mesh/ or nurbs/ folder, take out labels, and rewrite the zip.

    files: label -> the file, or files, to copy in for it, each with a suffix of the kind (.stl for mesh; .iges or
        .step for nurbs). They replace every file the label had.
    origin: where the files come from (see check_origin), one for all or one per label.
    remove: labels whose files are taken out.
    Returns the labels of this kind afterwards. A scan that has only the zip, as a downloaded copy does, is unpacked
    first.
    """
    if kind not in layout.MODEL_KINDS:
        raise ValueError(f"Unknown kind '{kind}'; the kinds are {list(layout.MODEL_KINDS)}.")
    folder = _scan_folder(scan)
    info = ScanInfo.load(folder / layout.SCAN_JSON)
    suffixes = layout.MODEL_KINDS[kind]
    remove = {check_label(label) for label in remove}
    sources: dict[str, list[Path]] = {}
    for label, paths in (files or {}).items():
        paths = [Path(paths)] if isinstance(paths, (str, os.PathLike)) else [Path(p) for p in paths]
        for path in paths:
            if path.suffix.lower() not in suffixes:
                raise ValueError(f"'{path}' cannot be a {kind} file, which is {' or '.join(suffixes)}.")
            if not path.is_file():
                raise FileNotFoundError(f"'{path}' does not exist.")
        sources[check_label(label)] = paths
    both = sorted(set(sources) & remove)
    if both:
        raise ValueError(f"{both} are both added and removed.")
    added = _new_origins(set(sources), origin, kind)

    model_folder = layout.model_folder(folder, kind)
    if not model_folder.is_dir() and layout.model_zip(folder, kind).is_file():
        unpack_scan(folder)
    model_folder.mkdir(exist_ok=True)
    for entry in model_folder_files(model_folder, kind):
        if entry.stem in remove or entry.stem in sources:
            entry.unlink()
    for label, paths in sources.items():
        for path in paths:
            shutil.copyfile(path, model_folder / f"{label}{path.suffix.lower()}")

    present = sorted({entry.stem for entry in model_folder_files(model_folder, kind)})
    if present:
        pack_scan(folder)
    else:
        shutil.rmtree(model_folder)
        layout.model_zip(folder, kind).unlink(missing_ok=True)
    _record(scan, folder, info, kind, present, added)
    return info.labels(kind)
