"""Changing a dataset in place: adding, replacing and removing the labels of a scan.

Whether a change is made in place or published as a new dataset is up to whoever maintains the dataset (see the
README). In place, the origin each label has in its file's JSON tells the labels of the source from those added later:
a bone added by a model is model:<name>@<version>, one drawn or corrected by hand is manual.

A scan holds numbered files of each kind (segmentation-01.nrrd, segmentation-02.nrrd, ...), each from one source. These
functions change one of them, `number`, or write a new one when no number is given: its JSON file names `source` (the
dataset's first source by default) and the scan's path (e.g. ds-0001/sub-000001/scan-01) as its source_scan_id.

These functions keep the JSON files in step with the data files; a file left without labels is deleted and the files
after it are numbered again. They leave dataset.json alone: when a label is now segmented in every scan of a source,
add it to its annotated_labels yourself. Rebuild the index afterwards.

    update_segmentation(scan, {"FEMUR_LEFT": mask}, origin="model:TotalSegmentator@2.4.0")  # a new segmentation file
    update_segmentation(scan, remove=["PATELLA_LEFT"], number=2)                     # in segmentation-02.nrrd
    update_model_files(scan, "mesh", {"FEMUR_LEFT": Path("femur.stl")}, origin="derived:my-mesher@1.0", number=1)
    update_model_files(scan, "mesh", remove=["FEMUR_LEFT"], number=1)

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
from .labelmap import LabelMap
from .metadata import DatasetInfo, LabeledFileInfo, check_label, check_origin
from .reader import Scan
from .segmentation_file import (
    SEGMENTATION_SUFFIX,
    Segment,
    _io_modules,
    full_region,
    read_segmentation,
    write_regions,
)

if TYPE_CHECKING:
    import SimpleITK as sitk

Origin = str | Mapping[str, str]


def _scan_folder(scan: Scan | Path) -> Path:
    folder = scan.path if isinstance(scan, Scan) else Path(scan)
    if not (folder / layout.SCAN_JSON).is_file():
        raise FileNotFoundError(f"'{folder}' has no {layout.SCAN_JSON}.")
    return folder


def _numbers(folder: Path, kind: str) -> list[int]:
    """The numbers of a scan's files of a kind."""
    numbers = set()
    for entry in folder.iterdir():
        parsed = layout.parse_scan_entry(entry.name)
        if parsed is not None and parsed[0] == kind:
            numbers.add(parsed[1])
    return sorted(numbers)


def _file(folder: Path, kind: str, number: int | None, source: int | None) -> tuple[int, LabeledFileInfo]:
    """The number of the file that is changed and its JSON contents: those of an existing file, or of a new one."""
    numbers = _numbers(folder, kind)
    if number is not None:
        if number not in numbers:
            raise ValueError(f"'{folder}' has no {kind}-{number:02d}; its {kind} files are {numbers}.")
        if source is not None:
            raise ValueError("source is given for a new file only; an existing file keeps its source.")
        return number, LabeledFileInfo.load(layout.data_json(folder, kind, number))
    if source is None:
        dataset_json = folder.parent.parent / layout.DATASET_JSON
        if not dataset_json.is_file():
            raise ValueError(f"'{folder}' is not in a dataset folder with a {layout.DATASET_JSON}; give source.")
        source = next(iter(DatasetInfo.load(dataset_json).sources))
    key = f"{folder.parent.parent.name}/{folder.parent.name}/{folder.name}"
    return (numbers[-1] + 1 if numbers else 1), LabeledFileInfo(source=source, source_scan_id=key)


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


def _renumber(folder: Path, kind: str) -> None:
    """Close the gaps in the numbers of a scan's files of a kind (after one was deleted)."""
    for new, old in enumerate(_numbers(folder, kind), start=1):
        if new == old:
            continue
        for entry in folder.iterdir():
            parsed = layout.parse_scan_entry(entry.name)
            if parsed is not None and parsed[0] == kind and parsed[1] == old:
                entry.rename(folder / entry.name.replace(f"{kind}-{old:02d}", f"{kind}-{new:02d}", 1))


def _record(scan, folder: Path, kind: str, number: int, info: LabeledFileInfo, present: Iterable[str], added: dict[str, str]):
    """Write a file's labels into its JSON file: the added ones with their new origin, the others as they were; delete
    the JSON file of a file left without labels."""
    origins = {}
    for label in present:
        if label in added:
            origins[label] = added[label]
        elif label in info.labels:
            origins[label] = info.labels[label]
        else:
            raise ValueError(f"'{folder}' has {kind} data for {label}, which its JSON file does not list; fix the scan first.")
    json_path = layout.data_json(folder, kind, number)
    if origins:
        info.set_labels(list(origins), origins)
        info.save(json_path)
    else:
        json_path.unlink(missing_ok=True)
        _renumber(folder, kind)
    if isinstance(scan, Scan):
        scan.__dict__.pop("info", None)  # read again when next used
    return list(info.labels) if origins else []


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
    masks: Mapping[LabelMap | str | Segment, sitk.Image] | Path | None = None,
    origin: Origin | None = None,
    remove: Iterable[str] = (),
    overlap: Literal["layer", "replace", "keep"] = "layer",
    number: int | None = None,
    source: int | None = None,
) -> list[str]:
    """Add or replace labels in a segmentation of the scan, and take out the labels in `remove`.

    masks: label (or Segment, for an instance or a rater) -> its region, a SimpleITK image on the grid of the scan's
        image (non-zero voxels are the label's); or the path of a BoneHub segmentation file, whose segments are all
        added. A label added replaces that label, all its segments included.
    origin: where the added labels come from (see check_origin), one for all or one per label.
    remove: labels to take out.
    overlap: where an added label covers a voxel of another label: 'layer' keeps both (overlapping segments go into
        separate layers); 'replace' gives the voxel to the new label; 'keep' leaves it to the label already there.
    number: the segmentation that is changed (segmentation-<number>.nrrd); a new one by default.
    source: the BoneHub ID of the source of a new segmentation; the dataset's first source by default.
    Returns the labels of the segmentation afterwards. A segmentation left without labels is deleted. The file is
    written again with its segments in the fewest layers.
    """
    np, sitk = _io_modules()
    folder = _scan_folder(scan)
    number, info = _file(folder, "segmentation", number, source)
    remove = {check_label(label) for label in remove}
    if overlap not in ("layer", "replace", "keep"):
        raise ValueError(f"overlap is 'layer', 'replace' or 'keep', not {overlap!r}.")

    def regions(layer_images, segments):
        """The region (see write_regions) of each segment of a file, made from its box in its layer when needed."""
        layers = []  # per layer: its image (which the array views) and each number's box
        for image in layer_images:
            shapes = sitk.LabelShapeStatisticsImageFilter()
            shapes.Execute(image)
            layers.append((image, sitk.GetArrayViewFromImage(image), {int(n): shapes.GetBoundingBox(n) for n in shapes.GetLabels()}))

        def region(segment):
            def make():
                _, array, boxes = layers[segment.layer]
                if segment.number not in boxes:
                    return None
                x, y, z, sx, sy, sz = boxes[segment.number]
                box = (slice(z, z + sz), slice(y, y + sy), slice(x, x + sx))
                return box, array[box] == segment.number

            return make

        return {Segment(s.label, instance=s.instance, rater=s.rater): region(s) for s in segments}

    def union(found) -> object:
        taken = np.zeros(tuple(reversed(grid.GetSize())), dtype=bool)
        for make in found.values():
            if (made := make()) is not None:
                taken[made[0]] |= made[1]
        return taken

    def without(make, taken):
        def made():
            if (region := make()) is None:
                return None
            box, mask = region
            return box, mask & ~taken[box]

        return made

    new_masks: dict[Segment, object] = {}
    grid_of_masks = None
    if masks is not None and not isinstance(masks, Mapping):
        layer_images, file_segments = read_segmentation(Path(masks))
        grid_of_masks = layer_images[0]
        new_masks = regions(layer_images, file_segments)
    elif masks:
        for key, image in masks.items():
            segment = key if isinstance(key, Segment) else Segment(check_label(key.name if isinstance(key, LabelMap) else key))
            if grid_of_masks is None:
                grid_of_masks = image
            _check_same_grid(image, grid_of_masks)
            new_masks[Segment(segment.label, instance=segment.instance, rater=segment.rater)] = full_region(
                sitk.GetArrayViewFromImage(image))

    segmentation_path = layout.data_file(folder, "segmentation", number)
    if segmentation_path.is_file():
        layer_images, segments = read_segmentation(segmentation_path)
        grid = layer_images[0]
        old_masks = regions(layer_images, segments)
    else:
        grid = _grid(folder, grid_of_masks)
        if grid is None:
            return []  # nothing to remove from, nothing to add
        old_masks = {}
    if grid_of_masks is not None:
        _check_same_grid(grid_of_masks, grid)

    added_names = {segment.label.name for segment in new_masks}
    both = sorted(added_names & remove)
    if both:
        raise ValueError(f"{both} are both added and removed.")
    added = _new_origins(added_names, origin, "segmentation")

    # every segment of the removed and the replaced labels goes
    kept = {s: make for s, make in old_masks.items() if s.label.name not in remove | added_names}
    if overlap == "replace" and new_masks:
        taken = union(new_masks)
        kept = {s: without(make, taken) for s, make in kept.items()}
    elif overlap == "keep" and kept:
        taken = union(kept)
        new_masks = {s: without(make, taken) for s, make in new_masks.items()}

    present = []
    if kept or new_masks:
        temporary = folder / f".{os.getpid()}.tmp{SEGMENTATION_SUFFIX}"
        try:
            present = write_regions({**kept, **new_masks}, grid, temporary)
            if present:
                os.replace(temporary, segmentation_path)
        finally:
            temporary.unlink(missing_ok=True)
    if not present:
        segmentation_path.unlink(missing_ok=True)
    return _record(scan, folder, "segmentation", number, info, present, added)


def update_model_files(
    scan: Scan | Path,
    kind: str,
    files: Mapping[str, Path | Sequence[Path]] | None = None,
    origin: Origin | None = None,
    remove: Iterable[str] = (),
    number: int | None = None,
    source: int | None = None,
) -> list[str]:
    """Add or replace the files of labels in a mesh or nurbs folder of the scan (e.g. mesh-01/), take out labels, and
    rewrite the zip.

    files: label -> the file, or files, to copy in for it, each with a suffix of the kind (.stl for mesh; .iges or
        .step for nurbs). They replace every file the label had.
    origin: where the files come from (see check_origin), one for all or one per label.
    remove: labels whose files are taken out.
    number: the folder that is changed (mesh-<number>/); a new one by default.
    source: the BoneHub ID of the source of a new folder; the dataset's first source by default.
    Returns the labels of the folder afterwards. A scan that has only the zip, as a downloaded copy does, is unpacked
    first.
    """
    if kind not in layout.MODEL_KINDS:
        raise ValueError(f"Unknown kind '{kind}'; the kinds are {list(layout.MODEL_KINDS)}.")
    folder = _scan_folder(scan)
    number, info = _file(folder, kind, number, source)
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

    model_folder = layout.model_folder(folder, kind, number)
    model_zip = layout.model_zip(folder, kind, number)
    if not model_folder.is_dir() and model_zip.is_file():
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
        model_zip.unlink(missing_ok=True)
    return _record(scan, folder, kind, number, info, present, added)
