"""Reading and writing BoneHub segmentation masks (`.nrrd` files with 3D Slicer segment fields in their header).

Voxels hold per-file segment numbers (1, 2, 3 ...; uint8, or uint16 above 255 segments), as in 3D Slicer's own
segmentation files; the header names the label of each number (`Segment<i>_Name`) and records the segment's colour and
bounding box. Segments that overlap lie in separate layers of one file: the file is then 4D, its first axis (kind
`list`) the layers, as 3D Slicer writes overlapping segments, and a segment's number is unique within its layer only.
One rule lays out every file (write_regions): the segments, in label map order, each go into the first layer where
they overlap nothing, and are numbered 1, 2, ... within it.

A label may have several segments in one file, told apart by their instance (lesion 1, lesion 2; the fragments of a
broken bone, the main fragment first) or rater (the readers of a dataset with several). Together they are the label's
region. Each segment's header tags hold BoneHubLabel and, where they apply, BoneHubInstance and BoneHubRater.

    write_segmentation({LabelMap.FEMUR_LEFT: mask, ...}, reference, path)   one binary mask per label (or Segment)
    write_regions({Segment(...): region, ...}, reference, path)             each segment's box, made when needed
    write_indexed_segmentation(numbers, segments, reference, path)          segment numbers already in the voxels
    read_segmentation(path)            the segment-number image of each layer, and the segments
    read_label_mask(path, label)       one label's region as a 0/1 image

Reading and writing masks requires numpy and SimpleITK: pip install "bonehub-format[io]".
The header functions (read_segmentation_labels, read_segmentation_header, read_segmentation_segments) need neither.
"""

from __future__ import annotations

import colorsys
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from .labelmap import LabelMap, node_of, part_of, side_of, tissue_of
from .layout import SEGMENTATION_SUFFIX

if TYPE_CHECKING:
    import numpy as np
    import SimpleITK as sitk

# Slices processed at a time when relabelling, to limit memory.
_CHUNK_SLICES = 32
_LABEL_ORDER = {label: i for i, label in enumerate(LabelMap)}


def _io_modules():
    # Imported on use, so the base install (without numpy/SimpleITK) can import this module.
    try:
        import numpy
        import SimpleITK
    except ImportError as e:
        raise ImportError(
            'Reading or writing BoneHub segmentation files needs numpy and SimpleITK: pip install "bonehub-format[io]"'
        ) from e
    return numpy, SimpleITK


def _label(label: LabelMap | str) -> LabelMap:
    if isinstance(label, LabelMap):
        return label
    try:
        return LabelMap[label.name if isinstance(label, Enum) else str(label)]
    except KeyError:
        raise ValueError(f"'{label}' is not a label of the label map (see bonehub_format/labelmap.py).") from None


@dataclass(frozen=True)
class Segment:
    """One segment of a segmentation file: its label, its layer and number within the layer, and its instance and
    rater where the file holds several segments of the label."""

    label: LabelMap
    layer: int = 0
    number: int = 1
    instance: int | None = None
    rater: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "label", _label(self.label))

    @property
    def id(self) -> str:
        """The segment's ID in the file, unique within it: the label name, then #instance and @rater."""
        return self.label.name + (f"#{self.instance}" if self.instance is not None else "") + (
            f"@{self.rater}" if self.rater is not None else "")

    @property
    def key(self) -> tuple:
        return (self.label, self.instance, self.rater)


def segment_number_dtype(number_of_segments: int):
    """The smallest unsigned type that holds segment numbers 1..number_of_segments."""
    np, _ = _io_modules()
    if number_of_segments <= 255:
        return np.uint8
    if number_of_segments <= 65535:
        return np.uint16
    raise ValueError(f"{number_of_segments} segments in one layer; at most 65535 are supported.")


def _segment_color(label: LabelMap) -> tuple[float, float, float]:
    """A stable colour per label: the hue from its structure, the saturation from its part and tissue, the lightness
    from its side (left lighter, right darker), so the two sides of a structure always differ."""
    hue = zlib.crc32(node_of(label).encode()) % 360 / 360
    saturation = 0.45 + zlib.crc32(f"{part_of(label)}|{tissue_of(label)}".encode()) % 4 * 0.12
    lightness = {"LEFT": 0.6, "RIGHT": 0.36}.get(side_of(label), 0.48)
    return colorsys.hls_to_rgb(hue, lightness, saturation)


def write_segmentation(masks: Mapping, reference_image: sitk.Image, output_path: Path) -> list[str]:
    """Write a mask given as one binary array per label.

    masks: label (a LabelMap member or its name) or Segment (for an instance or a rater) -> a boolean voxel array on the
        grid of reference_image (non-zero voxels are the label's). Masks that overlap go into separate layers (see
        write_regions).
    reference_image: image whose geometry (spacing, origin, direction) the mask shares.
    output_path: destination, ending in `.nrrd`.
    Returns the names of the labels present in the mask.
    """
    np, _ = _io_modules()
    shape = tuple(reversed(reference_image.GetSize()))
    regions = {}
    for key, array in masks.items():
        if np.shape(array) != shape:
            raise ValueError(f"The mask of {key} has shape {np.shape(array)}, not the grid's {shape} (z, y, x).")
        segment = key if isinstance(key, Segment) else Segment(key)
        regions[Segment(segment.label, instance=segment.instance, rater=segment.rater)] = full_region(array)
    return write_regions(regions, reference_image, output_path)


def _segment_order(segment: Segment) -> tuple:
    return (_LABEL_ORDER[segment.label], segment.rater or "", segment.instance or 0)


def bounding_box(mask) -> tuple[slice, ...] | None:
    """The bounding box of a boolean array's True voxels, as one slice per axis; None when it has none."""
    np, _ = _io_modules()
    box = []
    for axis in range(mask.ndim):
        hits = np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != axis)))
        if not hits.size:
            return None
        box.append(slice(int(hits[0]), int(hits[-1]) + 1))
    return tuple(box)


def full_region(array) -> Callable:
    """The region (see write_regions) of a mask given as a whole voxel array, its non-zero voxels the segment's."""

    def region():
        np, _ = _io_modules()
        mask = np.asarray(array)
        mask = mask if mask.dtype == bool else mask != 0
        box = bounding_box(mask)
        return None if box is None else (box, mask[box])

    return region


def write_regions(regions: Mapping[Segment, Callable], reference_image: sitk.Image, output_path: Path) -> list[str]:
    """Write segments given by their regions: how every BoneHub segmentation file is laid out.

    regions: segment (its label, instance and rater; its layer and number are chosen here) -> a function that returns
        the segment's region as (box, mask): the slices of a box in the voxel grid of reference_image (z, y, x) and a
        boolean array of the box's shape; or None when the segment has no voxels. The functions are called one at a
        time, so only one segment's box is in memory at once besides the layers.
    The segments go, in label map order (then rater, then instance), each into the first layer where it overlaps no
    segment already placed; a layer is added only when none has room. Within a layer they are numbered 1, 2, ... in
    that order. The same regions therefore always give the same file, whichever function writes it.
    The other arguments are as for write_segmentation.
    """
    layers, segments = _place_regions(regions, tuple(reversed(reference_image.GetSize())))
    return _write_layers(layers, segments, reference_image, output_path)


def _place_regions(regions: Mapping[Segment, Callable], shape: tuple[int, ...]) -> tuple[list, list[Segment]]:
    """The layers (arrays of segment numbers on a grid of `shape`) and the placed segments, as write_regions lays
    them out; for a writer that frees its sources before the layers are written (see _write_layers)."""
    np, _ = _io_modules()
    layers, segments = [], []
    for segment in sorted(regions, key=_segment_order):
        region = regions[segment]()
        if region is None or not region[1].any():
            continue
        box, mask = region
        if len(box) != 3 or any(s.stop > n for s, n in zip(box, shape)) or mask.shape != tuple(s.stop - s.start for s in box):
            raise ValueError(f"The region of {segment.id} does not lie in the grid {shape} (z, y, x).")
        layer = next((n for n, numbers in enumerate(layers) if not numbers[box][mask].any()), len(layers))
        if layer == len(layers):
            layers.append(np.zeros(shape, dtype=np.uint8))
        number = 1 + sum(1 for s in segments if s.layer == layer)
        if number > np.iinfo(layers[layer].dtype).max:
            layers[layer] = layers[layer].astype(segment_number_dtype(number))
        layers[layer][box][mask] = number
        segments.append(Segment(segment.label, layer, number, segment.instance, segment.rater))
    if not layers:
        layers = [np.zeros(shape, dtype=np.uint8)]
    return layers, segments


def _check_segments(segments: Sequence[Segment]) -> None:
    for segment in segments:
        if segment.number < 1:
            raise ValueError(f"{segment}: segment numbers start at 1.")
        if segment.instance is not None and segment.instance < 1:
            raise ValueError(f"{segment}: instances are numbered from 1.")
    keys = [s.key for s in segments]
    if len(set(keys)) != len(keys):
        raise ValueError("Each label, instance and rater may have only one segment in a file.")
    places = [(s.layer, s.number) for s in segments]
    if len(set(places)) != len(places):
        raise ValueError("Each segment number may be used once per layer.")
    for label, rater in {(s.label, s.rater) for s in segments}:
        same = [s for s in segments if (s.label, s.rater) == (label, rater)]
        if len(same) > 1 and any(s.instance is None for s in same):
            raise ValueError(f"{label.name}: several segments of a label need an instance number each.")


def write_indexed_segmentation(numbers, segments: Sequence, reference_image: sitk.Image, output_path: Path) -> list[str]:
    """Write a mask whose voxels already hold segment numbers.

    numbers: voxel array of segment numbers (0 for background), or a list of such arrays, one per layer.
    segments: the Segment of each number in each layer; or, for one layer, the label of each segment number (number k
        is segments[k - 1]).
    The other arguments are as for write_segmentation. Segments without voxels are left out of the header.
    """
    np, _ = _io_modules()
    layers = [np.asarray(a) for a in numbers] if isinstance(numbers, (list, tuple)) else [np.asarray(numbers)]
    return _write_layers(layers, segments, reference_image, output_path)


def _write_layers(layers: list, segments: Sequence, reference_image: sitk.Image, output_path: Path) -> list[str]:
    """write_indexed_segmentation, letting go of each array of `layers` (its entries become None) once the array is in
    the image to write, so that a writer that owns its layers (write_regions) does not hold them twice."""
    np, sitk = _io_modules()
    output_path = Path(output_path)
    if not output_path.name.endswith(SEGMENTATION_SUFFIX):
        raise ValueError(f"A BoneHub segmentation file must end in '{SEGMENTATION_SUFFIX}': {output_path}")
    segments = [s if isinstance(s, Segment) else Segment(s, 0, k) for k, s in enumerate(segments, start=1)]
    _check_segments(segments)
    if any(s.layer >= len(layers) for s in segments):
        raise ValueError(f"A segment lies in a layer the mask does not have ({len(layers)} layers).")

    # one voxel type for all layers, as a 4D file has one
    dtype = segment_number_dtype(max((s.number for s in segments), default=1))
    number_of_layers, images, extents = len(layers), [], {}
    for layer in range(number_of_layers):
        image = sitk.GetImageFromArray(np.asarray(layers[layer], dtype=dtype))
        layers[layer] = None
        image.CopyInformation(reference_image)
        boxes = sitk.LabelShapeStatisticsImageFilter()  # Slicer loads a segmentation faster with each extent known
        boxes.Execute(image)
        for number in boxes.GetLabels():
            x, y, z, sx, sy, sz = boxes.GetBoundingBox(number)
            extents[(layer, int(number))] = f"{x} {x + sx - 1} {y} {y + sy - 1} {z} {z + sz - 1}"
        images.append(image)
    out = images[0] if number_of_layers == 1 else sitk.Compose(images)
    images.clear()
    out.CopyInformation(reference_image)

    present = []
    order = sorted((s for s in segments if (s.layer, s.number) in extents),
                   key=_segment_order)
    for i, segment in enumerate(order):
        r, g, b = _segment_color(segment.label)
        tags = f"BoneHubLabel:{segment.label.name}|"
        if segment.instance is not None:
            tags += f"BoneHubInstance:{segment.instance}|"
        if segment.rater is not None:
            tags += f"BoneHubRater:{segment.rater}|"
        out.SetMetaData(f"Segment{i}_ID", segment.id)
        out.SetMetaData(f"Segment{i}_Name", segment.label.name)
        out.SetMetaData(f"Segment{i}_NameAutoGenerated", "0")
        out.SetMetaData(f"Segment{i}_Color", f"{r:.6f} {g:.6f} {b:.6f}")
        out.SetMetaData(f"Segment{i}_ColorAutoGenerated", "0")
        out.SetMetaData(f"Segment{i}_Extent", extents[(segment.layer, segment.number)])
        out.SetMetaData(f"Segment{i}_LabelValue", str(segment.number))
        out.SetMetaData(f"Segment{i}_Layer", str(segment.layer))
        out.SetMetaData(f"Segment{i}_Tags", tags)
        if segment.label.name not in present:
            present.append(segment.label.name)
    out.SetMetaData("Segmentation_MasterRepresentation", "Binary labelmap")
    out.SetMetaData("Segmentation_ContainedRepresentationNames", "Binary labelmap|")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(output_path))
    writer.UseCompressionOn()
    writer.SetCompressionLevel(9)
    writer.Execute(out)
    if number_of_layers > 1:
        _mark_layer_axis(output_path)
    return present


def _mark_layer_axis(path: Path) -> None:
    """ITK writes the layer axis of a 4D mask as kind 'vector'; 3D Slicer writes and expects 'list'."""
    data = Path(path).read_bytes()
    end = data.index(b"\n\n") + 2
    header = data[:end].replace(b"kinds: vector domain domain domain", b"kinds: list domain domain domain")
    Path(path).write_bytes(header + data[end:])


def _tags(image, i: int) -> dict[str, str]:
    key = f"Segment{i}_Tags"
    if not image.HasMetaDataKey(key):
        return {}
    return dict(tag.split(":", 1) for tag in image.GetMetaData(key).split("|") if ":" in tag)


def read_segmentation(path: Path) -> tuple[list[sitk.Image], list[Segment]]:
    """Read a mask as stored: the image of segment numbers of each layer (one for a file whose segments do not
    overlap), and every segment of the file, which says the label, layer and number of each."""
    _, sitk = _io_modules()
    image = sitk.ReadImage(str(path))
    segments = []
    i = 0
    while image.HasMetaDataKey(f"Segment{i}_LabelValue"):
        tags, name = _tags(image, i), image.GetMetaData(f"Segment{i}_Name")
        name = tags.get("BoneHubLabel", name)
        if name not in LabelMap.__members__:
            raise ValueError(f"Segment '{name}' in '{path}' is not a label of the label map.")
        layer = int(image.GetMetaData(f"Segment{i}_Layer")) if image.HasMetaDataKey(f"Segment{i}_Layer") else 0
        segments.append(Segment(LabelMap[name], layer, int(image.GetMetaData(f"Segment{i}_LabelValue")),
                                int(tags["BoneHubInstance"]) if "BoneHubInstance" in tags else None,
                                tags.get("BoneHubRater")))
        i += 1
    if image.GetNumberOfComponentsPerPixel() > 1:
        layers = [sitk.VectorIndexSelectionCast(image, k) for k in range(image.GetNumberOfComponentsPerPixel())]
    else:
        layers = [image]
    return layers, segments


def read_label_mask(path: Path, label: LabelMap | str, instance: int | None = None, rater: str | None = None) -> sitk.Image:
    """A label's region in a mask, as a 0/1 uint8 image: the union of its segments (only the given instance or rater,
    where given)."""
    np, sitk = _io_modules()
    label = _label(label)
    layers, segments = read_segmentation(path)
    region = None
    for s in segments:
        if s.label != label or (instance is not None and s.instance != instance) or (rater is not None and s.rater != rater):
            continue
        voxels = sitk.GetArrayViewFromImage(layers[s.layer]) == s.number
        region = voxels if region is None else region | voxels
    if region is None:
        region = np.zeros(sitk.GetArrayViewFromImage(layers[0]).shape, dtype=bool)
    out = sitk.GetImageFromArray(region.astype(np.uint8))
    out.CopyInformation(layers[0])
    return out


def read_segmentation_header(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """The fields ('sizes', 'type', ...) and the key/value pairs ('Segment0_Name', ...) of a segmentation header.

    Reads only the plain-text header at the start of the file, without numpy or SimpleITK.
    """
    fields, pairs = {}, {}
    with open(path, "rb") as f:
        if not f.readline().startswith(b"NRRD"):
            raise ValueError(f"'{path}' is not a NRRD file.")
        for raw_line in f:
            line = raw_line.decode("latin-1").rstrip("\r\n")
            if not line:  # the header ends at the first empty line
                break
            if line.startswith("#"):
                continue
            if ":=" in line:
                key, value = line.split(":=", 1)
                pairs[key] = value
            elif ": " in line:
                key, value = line.split(": ", 1)
                fields[key.lower()] = value
    return fields, pairs


def read_segmentation_segments(path: Path) -> tuple[tuple[int, ...], dict[str, tuple[int, ...] | None]]:
    """The size of a mask's voxel grid (x, y, z) and, per label, the extent of its segments in voxels (x_min x_max
    y_min y_max z_min z_max, as 3D Slicer writes it; their union where a label has several), read from the header
    only. The extent is None where the header has none."""
    fields, pairs = read_segmentation_header(path)
    sizes = tuple(int(size) for size in fields.get("sizes", "").split())
    if len(sizes) == 4:  # the first axis lists the layers
        sizes = sizes[1:]
    segments: dict[str, tuple[int, ...] | None] = {}
    i = 0
    while f"Segment{i}_Name" in pairs:
        name, extent = pairs[f"Segment{i}_Name"], pairs.get(f"Segment{i}_Extent")
        box = tuple(int(v) for v in extent.split()) if extent else None
        if name in segments:
            old = segments[name]
            box = None if old is None or box is None else tuple(
                min(old[k], box[k]) if k % 2 == 0 else max(old[k], box[k]) for k in range(6))
        segments[name] = box
        i += 1
    return sizes, segments


def read_segmentation_labels(path: Path) -> list[str]:
    """The names of the labels in a mask, from its header only (each once)."""
    return list(read_segmentation_segments(path)[1])
