import importlib.util
import tempfile
import unittest
from pathlib import Path

from bonehub_format import (
    BoneLabelMap,
    read_segmentation,
    read_segmentation_labels,
    read_segmentation_segments,
    write_indexed_segmentation,
    write_segmentation,
)

HAS_IO = all(importlib.util.find_spec(m) is not None for m in ("numpy", "SimpleITK"))


@unittest.skipUnless(HAS_IO, 'needs the io extra: pip install "bonehub-format[io]"')
class TestSegmentationFile(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import SimpleITK as sitk

        self.np, self.sitk = np, sitk
        self.tmp = Path(tempfile.mkdtemp())
        self.array = np.zeros((4, 8, 8), dtype=np.int32)
        self.array[0:2] = BoneLabelMap.FEMUR_LEFT.value
        self.array[2:4, :4] = BoneLabelMap.FEMUR_PROXIMAL_CORTICAL_LEFT.value
        self.reference = sitk.GetImageFromArray(np.zeros(self.array.shape, dtype=np.int16))

    def test_round_trip(self):
        path = self.tmp / "001_000001.seg.nrrd"
        written = write_segmentation(self.array, self.reference, path)

        self.assertEqual(written, ["FEMUR_LEFT", "FEMUR_PROXIMAL_CORTICAL_LEFT"])
        self.assertEqual(read_segmentation_labels(path), written)
        # the header alone gives the grid size (x, y, z) and each segment's extent, without SimpleITK
        sizes, segments = read_segmentation_segments(path)
        self.assertEqual(sizes, (8, 8, 4))
        self.assertEqual(segments, {"FEMUR_LEFT": (0, 7, 0, 7, 0, 1), "FEMUR_PROXIMAL_CORTICAL_LEFT": (0, 7, 0, 3, 2, 3)})
        bonehub = self.sitk.GetArrayFromImage(read_segmentation(path))
        self.assertTrue(self.np.array_equal(bonehub, self.array))

    def test_voxels_hold_small_segment_numbers(self):
        path = self.tmp / "numbers.seg.nrrd"
        write_segmentation(self.array, self.reference, path)
        image = self.sitk.ReadImage(str(path))
        self.assertEqual(image.GetPixelID(), self.sitk.sitkUInt8)
        self.assertEqual(sorted(self.np.unique(self.sitk.GetArrayViewFromImage(image))), [0, 1, 2])
        header = {image.GetMetaData(f"Segment{i}_Name"): image.GetMetaData(f"Segment{i}_LabelValue") for i in range(2)}
        self.assertEqual(header, {"FEMUR_LEFT": "1", "FEMUR_PROXIMAL_CORTICAL_LEFT": "2"})
        # bounding boxes Slicer reads as x_min x_max y_min y_max z_min z_max
        extents = {image.GetMetaData(f"Segment{i}_Name"): image.GetMetaData(f"Segment{i}_Extent") for i in range(2)}
        self.assertEqual(extents, {"FEMUR_LEFT": "0 7 0 7 0 1", "FEMUR_PROXIMAL_CORTICAL_LEFT": "0 7 0 3 2 3"})

    def test_more_than_255_segments_use_uint16(self):
        values = [label.value for label in BoneLabelMap if label.value][:300]
        array = self.np.zeros((1, 20, 20), dtype=self.np.int32)
        array.flat[: len(values)] = values
        path = self.tmp / "many.seg.nrrd"
        write_segmentation(array, self.sitk.GetImageFromArray(array), path)
        self.assertEqual(self.sitk.ReadImage(str(path)).GetPixelID(), self.sitk.sitkUInt16)
        self.assertTrue(self.np.array_equal(self.sitk.GetArrayFromImage(read_segmentation(path)), array))

    def test_indexed_writer_skips_numbers_not_in_the_mask(self):
        numbers = self.np.zeros((2, 4, 4), dtype=self.np.uint8)
        numbers[0] = 2
        values = [BoneLabelMap.TIBIA_LEFT.value, BoneLabelMap.FEMUR_LEFT.value]
        path = self.tmp / "indexed.seg.nrrd"
        written = write_indexed_segmentation(numbers, values, self.sitk.GetImageFromArray(numbers), path)
        self.assertEqual(written, ["FEMUR_LEFT"])
        with self.assertRaises(ValueError):
            write_indexed_segmentation(numbers, values * 2, self.sitk.GetImageFromArray(numbers), path)

    def test_wrong_suffix_is_rejected(self):
        with self.assertRaises(ValueError):
            write_segmentation(self.array, self.reference, self.tmp / "001_000002.nii.gz")
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_segment_added_by_hand_in_slicer(self):
        # no BoneHub tags: the label comes from the segment name
        image = self.sitk.GetImageFromArray((self.array > 0).astype(self.np.uint8))
        image.SetMetaData("Segment0_Name", "FEMUR_LEFT")
        image.SetMetaData("Segment0_LabelValue", "1")
        path = self.tmp / "manual.seg.nrrd"
        self.sitk.WriteImage(image, str(path))
        self.assertEqual(read_segmentation_labels(path), ["FEMUR_LEFT"])
        bonehub = self.sitk.GetArrayFromImage(read_segmentation(path))
        self.assertEqual(set(self.np.unique(bonehub)), {0, BoneLabelMap.FEMUR_LEFT.value})


if __name__ == "__main__":
    unittest.main()
