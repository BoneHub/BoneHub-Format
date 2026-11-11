"""Helpers for converters: reading source images and writing BoneHub images and segmentations.

Requires the converter extra: pip install "bonehub-format[converter]".
"""

import os
import re
from pathlib import Path

import highdicom
import numpy as np
import pydicom
import SimpleITK as sitk

from .segmentation_file import segment_number_dtype, write_indexed_segmentation

# Slices checked at a time, to limit memory.
_CHUNK_SLICES = 32
_INT16 = np.iinfo(np.int16)


def read_image(input_path: Path) -> sitk.Image:
    """An image file, or the first DICOM series of a folder.

    A NIfTI image whose direction cosines are slightly off orthonormal, which ITK refuses to read, is read with the
    nearest orthonormal directions instead (see read_nifti_orthonormalized).
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
        return read_nifti_orthonormalized(input_path)


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
        raise ValueError(f"The directions of '{input_path}' are too far from orthonormal to correct.")
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


def write_image(image: sitk.Image, output_path: Path) -> None:
    """Write an image as BoneHub stores it: compressed, in the pixel type compact_image chooses."""
    sitk.WriteImage(compact_image(image), str(output_path), useCompression=True)


def export_image(input_path: Path, output_path: Path) -> None:
    """Convert an image file or DICOM folder (see read_image) to `output_path` (see write_image)."""
    write_image(read_image(input_path), output_path)


def export_nii_segmentation(
    input_label_paths: list[Path],
    output_label_path: Path,
    label_mappings: list[dict],
):
    """
    Converts one or more NIfTI segmentation files to BoneHub standardized labels and saves the result.
    input_label_paths: list of Paths to the original label file(s) in NIfTI format.
    output_label_path: Path to save the combined label file; written as `.seg.nrrd`.
    label_mappings: list of dictionaries mapping original labels to BoneHub labels.
                    When multiple files are given, later files take priority over earlier ones in case of overlapping voxels.
    """
    if len(input_label_paths) != len(label_mappings):
        raise ValueError("The number of input label paths must match the number of label mappings.")

    ref_image = None
    # (image, label array, mapping) on the reference grid; the image is kept alive because
    # the array is a view of its memory.
    sources = []
    for input_label_path, label_mapping in zip(input_label_paths, label_mappings):
        image = sitk.ReadImage(str(input_label_path))
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

    # Number the BoneHub labels present 1..N and map each source label straight to its number.
    present = set()
    for _, array, label_mapping in sources:
        present.update(label_mapping[v] for v in _unique_labels(array) if label_mapping.get(v))
    bonehub_values = sorted(present)
    number_of = {value: n for n, value in enumerate(bonehub_values, start=1)}
    dtype = segment_number_dtype(len(bonehub_values))

    numbers = None
    for _, array, label_mapping in sources:
        lookup = np.zeros(max(int(array.max()), max(label_mapping)) + 1, dtype=dtype)
        for orig_label, bonehub_label in label_mapping.items():
            if bonehub_label in number_of:
                lookup[orig_label] = number_of[bonehub_label]
        mapped = lookup[array]
        # Non-zero voxels from later files overwrite earlier ones
        if numbers is None:
            numbers = mapped
        else:
            np.copyto(numbers, mapped, where=mapped != 0)
        del mapped

    return write_indexed_segmentation(numbers, bonehub_values, ref_image, output_label_path)


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
    output_label_path: Path to save the converted label file; written as `.seg.nrrd`.
    label_mapping: Dictionary mapping original labels to BoneHub labels.
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
    bonehub_values = sorted(set(bonehub_of.values()))
    number_of = {value: n for n, value in enumerate(bonehub_values, start=1)}
    lookup = np.zeros(max(int(seg_array.max()), max(bonehub_of)) + 1, dtype=segment_number_dtype(len(bonehub_values)))
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
        sitk.GetArrayViewFromImage(seg_resampled), bonehub_values, seg_resampled, output_label_path
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
