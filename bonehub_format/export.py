"""Helpers for converters: reading source images and writing BoneHub images and segmentations.

Requires the converter extra: pip install "bonehub-format[converter]".
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import highdicom
import numpy as np
import pydicom
import SimpleITK as sitk

from .labelmap import LabelMap
from .segmentation_file import Segment, segment_number_dtype, write_indexed_segmentation

# Slices checked at a time, to limit memory.
_CHUNK_SLICES = 32
_INT16 = np.iinfo(np.int16)
_LABEL_ORDER = {label: i for i, label in enumerate(LabelMap)}


class ShearedImageError(ValueError):
    """A NIfTI image whose axes are not perpendicular, beyond header rounding (a sheared grid)."""


def read_image(input_path: Path, sheared: int | None = None) -> sitk.Image:
    """An image file, or the first DICOM series of a folder.

    A NIfTI image whose direction cosines are slightly off orthonormal, which ITK refuses to read, is read with the
    nearest orthonormal directions instead (see read_nifti_orthonormalized). One whose axes are clearly not
    perpendicular (a sheared grid) raises ShearedImageError, unless `sheared` gives the SimpleITK interpolator to
    resample it with onto a grid with perpendicular axes (see read_nifti_resampled): sitk.sitkLinear for an image,
    sitk.sitkNearestNeighbor for a mask.
    """
    if Path(input_path).is_dir():
        series_reader = sitk.ImageSeriesReader()
        series_files = series_reader.GetGDCMSeriesFileNames(str(input_path))
        if not series_files:
            raise FileNotFoundError(f"No DICOM series found in '{input_path}'.")
        series_reader.SetFileNames(series_files)
        return series_reader.Execute()
    try:
        return sitk.ReadImage(str(input_path))
    except RuntimeError as e:
        if "orthonormal" not in str(e) or not str(input_path).lower().endswith((".nii", ".nii.gz")):
            raise
    try:
        return read_nifti_orthonormalized(input_path)
    except ShearedImageError:
        if sheared is None:
            raise
        return read_nifti_resampled(input_path, sheared)


_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0])
#: Largest deviation from orthonormal directions that read_nifti_orthonormalized corrects.
MAX_DIRECTION_ERROR = 1e-2


def read_nifti_orthonormalized(input_path: Path) -> sitk.Image:
    """A 3D NIfTI image with the nearest orthonormal direction cosines to those of its header.

    Some sources (e.g. TotalSegmentator) store an sform whose directions are off orthonormal by rounding (about 2e-4),
    and no qform; ITK refuses to read such a file. Here the voxel spacing is the length of each column of the affine,
    and the directions are the nearest orthonormal matrix to the normalised columns, which moves no voxel by more
    than a fraction of a millimetre for directions this close.
    """
    import nibabel as nib

    nifti = nib.load(str(input_path))
    if len(nifti.shape) != 3:
        raise ValueError(f"'{input_path}' is not a 3D image.")
    affine = nifti.affine
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    directions = affine[:3, :3] / spacing
    u, _, vt = np.linalg.svd(directions)
    orthonormal = u @ vt
    if np.abs(orthonormal - directions).max() > MAX_DIRECTION_ERROR:
        raise ShearedImageError(f"The axes of '{input_path}' are not perpendicular (a sheared grid); see read_nifti_resampled.")
    image = sitk.GetImageFromArray(np.asarray(nifti.dataobj).transpose())  # NIfTI is x-fastest, like ITK
    image.SetSpacing([float(s) for s in spacing])
    image.SetOrigin([float(v) for v in _RAS_TO_LPS @ affine[:3, 3]])
    image.SetDirection([float(v) for v in (_RAS_TO_LPS @ orthonormal).flatten()])
    return image


def _fits_int16(array: np.ndarray, integers_only: bool) -> bool:
    for start in range(0, array.shape[0], _CHUNK_SLICES):
        chunk = array[start : start + _CHUNK_SLICES]
        low, high = chunk.min(), chunk.max()
        if not (np.isfinite(low) and np.isfinite(high)) or low < _INT16.min or high > _INT16.max:
            return False
        if integers_only and not np.array_equal(chunk, np.rint(chunk)):
            return False
    return True


def compact_image(image: sitk.Image) -> sitk.Image:
    """The image in a pixel type that holds its values exactly and takes the least space.

    Floating point images whose values are all integers in the int16 range (as CT values in HU are) become int16, and
    so do integer images wider than 16 bits whose values fit; other float64 images become float32. The rest is kept.
    """
    if image.GetNumberOfComponentsPerPixel() != 1:
        return image
    array = sitk.GetArrayViewFromImage(image)
    if array.size == 0:
        return image
    if np.issubdtype(array.dtype, np.floating):
        if _fits_int16(array, integers_only=True):
            return sitk.Cast(image, sitk.sitkInt16)
        if array.dtype == np.float64:
            return sitk.Cast(image, sitk.sitkFloat32)
    elif np.issubdtype(array.dtype, np.integer) and array.dtype.itemsize > 2 and _fits_int16(array, integers_only=False):
        return sitk.Cast(image, sitk.sitkInt16)
    return image


def nifti_shear(input_path: Path) -> float:
    """How far the axes of a NIfTI image are from perpendicular, as the largest deviation of the angle between two
    axes from 90 degrees, from its header alone."""
    import nibabel as nib

    matrix = nib.load(str(input_path)).affine[:3, :3]
    directions = matrix / np.linalg.norm(matrix, axis=0)
    cosines = [abs(float(directions[:, i] @ directions[:, j])) for i, j in ((0, 1), (0, 2), (1, 2))]
    return float(np.degrees(np.arcsin(min(max(cosines), 1.0))))


def read_nifti_resampled(input_path: Path, interpolator: int) -> sitk.Image:
    """A 3D NIfTI image whose axes are not perpendicular (a sheared grid, as some acquisitions are), resampled onto a
    grid with perpendicular axes: the nearest orthonormal directions to the source's, the source's voxel spacing, and
    an extent that covers the whole source volume. Voxels outside the source are 0. `interpolator` is a SimpleITK
    interpolator: sitk.sitkLinear for an image (written as float32), sitk.sitkNearestNeighbor for a mask (its pixel
    type is kept). The image and the masks of a scan get the same grid, as they share the source's header."""
    import nibabel as nib

    nifti = nib.load(str(input_path))
    if len(nifti.shape) != 3:
        raise ValueError(f"'{input_path}' is not a 3D image.")
    matrix = _RAS_TO_LPS @ nifti.affine[:3, :3]
    offset = _RAS_TO_LPS @ nifti.affine[:3, 3]
    spacing = np.linalg.norm(matrix, axis=0)
    u, _, vt = np.linalg.svd(matrix / spacing)
    directions = u @ vt
    shape = np.array(nifti.shape[:3])
    corners = np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)], float)
    along_axes = (corners @ matrix.T + offset) @ directions  # the corners in the coordinates of the new axes
    low, high = along_axes.min(axis=0), along_axes.max(axis=0)
    size = (np.ceil((high - low) / spacing - 1e-6).astype(int) + 1).tolist()
    # the source in its index space (spacing 1, origin 0); the transform maps a point of the new grid to its index
    source = sitk.GetImageFromArray(np.asarray(nifti.dataobj).transpose())
    inverse = np.linalg.inv(matrix)
    to_index = sitk.AffineTransform(3)
    to_index.SetMatrix(inverse.ravel().tolist())
    to_index.SetTranslation((-inverse @ offset).tolist())
    pixel = sitk.sitkFloat32 if interpolator != sitk.sitkNearestNeighbor else source.GetPixelID()
    return sitk.Resample(
        source, size, to_index, interpolator, (directions @ low).tolist(), spacing.tolist(), directions.ravel().tolist(), 0.0, pixel
    )


def write_image(image: sitk.Image, output_path: Path) -> None:
    """Write an image as BoneHub stores it: compressed, in the pixel type compact_image chooses."""
    sitk.WriteImage(compact_image(image), str(output_path), useCompression=True)


def export_image(input_path: Path, output_path: Path, sheared: int | None = None) -> None:
    """Convert an image file or DICOM folder (see read_image, and its `sheared`) to `output_path` (see write_image)."""
    write_image(read_image(input_path, sheared), output_path)


@dataclass(frozen=True)
class Instance:
    """In a label mapping: a source label that is one of several segments of a label (lesion k; fragment k of a
    broken bone, the main fragment first)."""

    label: LabelMap
    instance: int


def _targets(target) -> list[tuple[LabelMap, int | None]]:
    """A mapping's target (a label, an Instance, or a list of them) as (label, instance) pairs; None is left out."""
    items = target if isinstance(target, (list, tuple)) else [target]
    out = []
    for item in items:
        if isinstance(item, Instance):
            out.append((Segment(item.label).label, int(item.instance)))
        elif item is not None:
            out.append((Segment(item).label, None))
    return out


def mapped_labels(*label_mappings: dict) -> list[str]:
    """The names of the labels that label mappings (see export_nii_segmentation) map to, in label map order: for a
    dataset's annotated_labels."""
    labels = {label for mapping in label_mappings for target in mapping.values() for label, _ in _targets(target)}
    return [label.name for label in sorted(labels, key=_LABEL_ORDER.__getitem__)]


def export_nii_segmentation(
    input_label_paths: list[Path],
    output_label_path: Path,
    label_mappings: list[dict],
):
    """
    Converts one or more label files (NIfTI, or anything SimpleITK reads) to BoneHub labels in one segmentation file.

    input_label_paths: the source label files; files on another grid are resampled (nearest neighbour) onto the first.
    output_label_path: the BoneHub segmentation (`.nrrd`) to write.
    label_mappings: per file, source label -> its target: a LabelMap label; an Instance(label, k), when the source label
        is one of several segments of a label (lesion k); or a list of these, when the source label's voxels belong to
        several labels at once (to build a whole from its pieces: KiTS's tumor -> [KIDNEY, KIDNEY_TUMOR]). Source
        labels mapped to the same target are merged into one segment, from one file or several (to build a whole from
        pieces in separate files: TotalSegmentator's heart.nii.gz and atrial_appendage_left.nii.gz -> HEART); source
        labels not in the mapping are left out.

    Nothing is overwritten: where labels overlap (two files, or a whole and its pieces), they go into separate layers of
    the file. The labels of one file at one position of the target lists form a group, and a target from several files
    or positions a group of its own; each group goes whole into the first layer it does not overlap.
    """
    if len(input_label_paths) != len(label_mappings):
        raise ValueError("The number of input label paths must match the number of label mappings.")

    ref_image = None
    # (image, label array, mapping) on the reference grid; the image is kept alive because
    # the array is a view of its memory.
    sources = []
    for input_label_path, label_mapping in zip(input_label_paths, label_mappings):
        image = read_image(input_label_path)  # TotalSegmentator and CADS files may need their directions orthonormalised
        if ref_image is None:
            ref_image = image
        elif image.GetSize() != ref_image.GetSize() or image.GetOrigin() != ref_image.GetOrigin():
            # Resample to match the reference image grid
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(ref_image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            image = resampler.Execute(image)
        sources.append((image, _label_array(image, input_label_path), label_mapping))

    # Groups of (file, {source label: target}) parts. The source labels of one file at one position of the target
    # lists form a group, which has no overlaps: the source labels of one file never share a voxel. A target that comes
    # from several files or positions (a whole built from pieces in separate files) is a group of its own, its voxels
    # the union of its parts'.
    found = [(i, source_label, k, target) for i, (_, array, label_mapping) in enumerate(sources)
             for source_label in sorted(_unique_labels(array) - {0})
             for k, target in enumerate(_targets(label_mapping.get(source_label)))]
    places: dict[tuple, set[tuple[int, int]]] = {}
    for i, _, k, target in found:
        places.setdefault(target, set()).add((i, k))
    groups: dict[tuple, dict[int, dict[int, tuple]]] = {}
    for i, source_label, k, target in found:
        group = ("file", i, k) if len(places[target]) == 1 else ("target", target)
        groups.setdefault(group, {}).setdefault(i, {})[source_label] = target

    def mask_of(parts: dict[int, dict[int, tuple]]):
        mask = None
        for i, mapping in parts.items():
            part = np.isin(sources[i][1], list(mapping))
            mask = part if mask is None else mask | part
        return mask

    # Each group into the first layer it does not overlap.
    occupied: list = []  # per layer, the voxels taken
    layer_of: dict[tuple, int] = {}
    for group, parts in groups.items():
        mask = mask_of(parts)
        layer = next((n for n, taken in enumerate(occupied) if not (taken & mask).any()), len(occupied))
        if layer == len(occupied):
            occupied.append(np.zeros(mask.shape, dtype=bool))
        occupied[layer] |= mask
        layer_of[group] = layer
    del occupied

    # Number the segments of each layer in label map order, and write each group's voxels into its layer.
    segments, number_of, layers = [], {}, []
    for layer in sorted(set(layer_of.values())):
        parts = [(i, mapping) for group, by_file in groups.items() if layer_of[group] == layer for i, mapping in by_file.items()]
        targets = sorted({t for _, mapping in parts for t in mapping.values()}, key=lambda t: (_LABEL_ORDER[t[0]], t[1] or 0))
        for number, (label, instance) in enumerate(targets, start=1):
            number_of[(label, instance)] = number
            segments.append(Segment(label, layer, number, instance))
        numbers = np.zeros(sources[0][1].shape, dtype=segment_number_dtype(len(targets)))
        for i, mapping in parts:
            array = sources[i][1]
            lookup = np.zeros(max(int(array.max()), max(mapping)) + 1, dtype=numbers.dtype)
            for source_label, target in mapping.items():
                lookup[source_label] = number_of[target]
            mapped = lookup[array]
            np.copyto(numbers, mapped, where=mapped != 0)
            del mapped
        layers.append(numbers)
    if not layers:
        layers = [np.zeros(sources[0][1].shape, dtype=np.uint8)]
    return write_indexed_segmentation(layers, segments, ref_image, output_label_path)


def _label_array(image: sitk.Image, path: Path) -> np.ndarray:
    """The image's labels as a non-negative integer array (a view when already integer)."""
    array = sitk.GetArrayViewFromImage(image)
    if not np.issubdtype(array.dtype, np.integer):  # some sources store masks as floats
        array = np.rint(array).astype(np.int32)
    if array.min() < 0:
        raise ValueError(f"Negative label values in '{path}'.")
    return array


def _unique_labels(array: np.ndarray, slices: int = 32) -> set[int]:
    """Distinct values in a large array, computed a block of slices at a time."""
    return {int(v) for z in range(0, array.shape[0], slices) for v in np.unique(array[z : z + slices])}


def _read_dicom_seg(path: Path) -> tuple[sitk.Image, dict]:
    """
    A DICOM SEG as an image of its segment numbers (0 for background), and the description of each segment by
    segment number. Overlapping segments raise an error.
    """
    seg = highdicom.seg.segread(path)
    volume = seg.get_volume(combine_segments=True)
    # The volume's array axes (0, 1, 2) are SimpleITK's (z, y, x).
    image = sitk.GetImageFromArray(volume.array)
    image.SetOrigin(volume.position)
    image.SetSpacing(volume.spacing[::-1])
    image.SetDirection(volume.direction[:, ::-1].flatten().tolist())
    return image, {int(item.SegmentNumber): item for item in seg.SegmentSequence}


def export_dicom_segmentation(
    input_image_path: Path,
    input_label_path: Path,
    output_label_path: Path,
    label_mapping: dict,
    dicom_segment_key: str = "SegmentLabel",
):
    """
    Converts original DICOM labels to BoneHub standardized labels and saves the result.
    input_image_path: Path to the original DICOM image folder.
    input_label_path: Path to the original DICOM label file.
    output_label_path: Path to save the converted label file; written as a BoneHub segmentation (`.nrrd`).
    label_mapping: Dictionary mapping the DICOM segments' labels to BoneHub labels (LabelMap members).
    dicom_segment_key: Key to access the segment label in the DICOM segmentation file. Options: "SegmentLabel" (default) or "SegmentDescription", depending on how the original labels are stored in the DICOM file.
    """
    seg_image, segment_infos = _read_dicom_seg(input_label_path)
    seg_array = _label_array(seg_image, input_label_path)

    # Map each DICOM segment to a BoneHub segment number 1..N
    bonehub_of = {}
    for orig_label, info in segment_infos.items():
        seg_label_name = getattr(info, dicom_segment_key, None)
        if seg_label_name is None:
            raise ValueError(f"Segment info for label {orig_label} does not contain key '{dicom_segment_key}'.")
        bonehub_of[orig_label] = label_mapping[seg_label_name]
    bonehub_labels = sorted({Segment(label).label for label in bonehub_of.values()}, key=_LABEL_ORDER.get)
    number_of = {label: n for n, label in enumerate(bonehub_labels, start=1)}
    bonehub_of = {orig: Segment(label).label for orig, label in bonehub_of.items()}
    lookup = np.zeros(max(int(seg_array.max()), max(bonehub_of)) + 1, dtype=segment_number_dtype(len(bonehub_labels)))
    for orig_label, bonehub_label in bonehub_of.items():
        lookup[orig_label] = number_of[bonehub_label]

    seg_image_mapped = sitk.GetImageFromArray(lookup[seg_array])
    seg_image_mapped.CopyInformation(seg_image)

    # Read the reference image series with SimpleITK (preserves LPS orientation)
    ref_image = read_image(input_image_path)

    # Resample segmentation directly in physical space to match the reference image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    seg_resampled = resampler.Execute(seg_image_mapped)

    return write_indexed_segmentation(
        sitk.GetArrayViewFromImage(seg_resampled), bonehub_labels, seg_resampled, output_label_path
    )


_DICOM_AGE = re.compile(r"(\d{1,3})([DWMY]?)")
_DAYS_PER_UNIT = {"D": 1, "W": 7, "M": 365.25 / 12, "Y": 365.25, "": 365.25}


def _dicom_age(value: str | None) -> int | None:
    """Age in years from a DICOM age string such as '065Y' or '018M'."""
    match = _DICOM_AGE.fullmatch((value or "").strip().upper())
    if not match:
        return None
    return int(int(match.group(1)) * _DAYS_PER_UNIT[match.group(2)] // 365.25)


def get_dicom_metadata(dicom_folder: Path) -> dict:
    """Age (years), gender (M, F, O or None) and modality from the header of the first file of a DICOM folder."""
    first_file = next(f for f in os.listdir(dicom_folder) if f.endswith(".dcm") or f.endswith(".dicom"))
    ds = pydicom.dcmread(os.path.join(dicom_folder, first_file), stop_before_pixels=True)
    gender = (getattr(ds, "PatientSex", None) or "").strip().upper() or None
    if gender not in (None, "M", "F", "O"):
        print(f"Warning: Unrecognized gender value '{gender}' in DICOM data folder '{dicom_folder}'. Setting gender to None.")
        gender = None
    return {
        "age": _dicom_age(getattr(ds, "PatientAge", None)),
        "gender": gender,
        "modality": getattr(ds, "Modality", None) or None,
    }


def write_stl(vertices: np.ndarray, faces: np.ndarray, output_path: Path) -> None:
    """Write a triangle mesh as a binary STL file.

    vertices: (N, 3) coordinates in millimetres (LPS, in the physical space of the scan's image where it has one).
    faces: (M, 3) vertex indices, starting at 0.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"A mesh has (N, 3) vertices and (M, 3) faces, not {vertices.shape} and {faces.shape}.")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("A face refers to a vertex that does not exist.")
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    records = np.zeros(len(faces), dtype=[("normal", "<f4", 3), ("triangle", "<f4", (3, 3)), ("attribute", "<u2")])
    records["normal"] = normals
    records["triangle"] = triangles
    with open(output_path, "wb") as f:
        f.write(b"BoneHub-Format binary STL".ljust(80, b" "))
        f.write(np.uint32(len(faces)).tobytes())
        f.write(records.tobytes())
