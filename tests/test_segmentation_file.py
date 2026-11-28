import importlib.util
import tempfile
import unittest
from pathlib import Path

from bonehub_format import (
    LabelMap,
    Segment,
    read_label_mask,
    read_segmentation,
    read_segmentation_header,
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
        self.femur = np.zeros((4, 8, 8), dtype=bool)
        self.femur[0:2] = True
        self.cortical = np.zeros_like(self.femur)
        self.cortical[2:4, :4] = True
        self.masks = {LabelMap.FEMUR_LEFT: self.femur, "FEMUR_PROXIMAL_CORTICAL_LEFT": self.cortical}
        self.reference = sitk.GetImageFromArray(np.zeros(self.femur.shape, dtype=np.int16))

    def test_round_trip(self):
        path = self.tmp / "001_000001.seg.nrrd"
        written = write_segmentation(self.masks, self.reference, path)

        self.assertEqual(written, ["FEMUR_LEFT", "FEMUR_PROXIMAL_CORTICAL_LEFT"])
        self.assertEqual(read_segmentation_labels(path), written)
        # the header alone gives the grid size (x, y, z) and each segment's extent, without SimpleITK
        sizes, segments = read_segmentation_segments(path)
        self.assertEqual(sizes, (8, 8, 4))
        self.assertEqual(segments, {"FEMUR_LEFT": (0, 7, 0, 7, 0, 1), "FEMUR_PROXIMAL_CORTICAL_LEFT": (0, 7, 0, 3, 2, 3)})
        layers, segments = read_segmentation(path)
        self.assertEqual(len(layers), 1)  # nothing overlaps: one layer
        self.assertEqual([(s.label, s.layer, s.number) for s in segments],
                         [(LabelMap.FEMUR_LEFT, 0, 1), (LabelMap.FEMUR_PROXIMAL_CORTICAL_LEFT, 0, 2)])
        mask = self.sitk.GetArrayFromImage(read_label_mask(path, "FEMUR_PROXIMAL_CORTICAL_LEFT"))
        self.assertTrue(self.np.array_equal(mask.astype(bool), self.cortical))

    def test_voxels_hold_small_segment_numbers(self):
        path = self.tmp / "numbers.seg.nrrd"
        write_segmentation(self.masks, self.reference, path)
        image = self.sitk.ReadImage(str(path))
        self.assertEqual(image.GetPixelID(), self.sitk.sitkUInt8)
        self.assertEqual(sorted(self.np.unique(self.sitk.GetArrayViewFromImage(image))), [0, 1, 2])
        header = {image.GetMetaData(f"Segment{i}_Name"): image.GetMetaData(f"Segment{i}_LabelValue") for i in range(2)}
        self.assertEqual(header, {"FEMUR_LEFT": "1", "FEMUR_PROXIMAL_CORTICAL_LEFT": "2"})
        # bounding boxes Slicer reads as x_min x_max y_min y_max z_min z_max
        extents = {image.GetMetaData(f"Segment{i}_Name"): image.GetMetaData(f"Segment{i}_Extent") for i in range(2)}
        self.assertEqual(extents, {"FEMUR_LEFT": "0 7 0 7 0 1", "FEMUR_PROXIMAL_CORTICAL_LEFT": "0 7 0 3 2 3"})

    def test_overlapping_labels_go_into_layers(self):
        kidney = self.np.zeros((4, 8, 8), dtype=bool)
        kidney[:, :6] = True
        cyst = self.np.zeros_like(kidney)
        cyst[:, 4:6] = True  # inside the kidney: a label means the whole structure, its cyst included
        path = self.tmp / "layers.seg.nrrd"
        written = write_segmentation({"KIDNEY_LEFT": kidney, "KIDNEY_CYST_LEFT": cyst}, self.reference, path)
        self.assertEqual(written, ["KIDNEY_LEFT", "KIDNEY_CYST_LEFT"])
        fields, _ = read_segmentation_header(path)
        self.assertEqual(fields["kinds"], "list domain domain domain")  # as 3D Slicer writes overlapping segments
        self.assertEqual(fields["sizes"], "2 8 8 4")
        self.assertEqual(read_segmentation_segments(path)[0], (8, 8, 4))
        layers, segments = read_segmentation(path)
        self.assertEqual([(s.label.name, s.layer) for s in segments], [("KIDNEY_LEFT", 0), ("KIDNEY_CYST_LEFT", 1)])
        for name, expected in (("KIDNEY_LEFT", kidney), ("KIDNEY_CYST_LEFT", cyst)):
            mask = self.sitk.GetArrayFromImage(read_label_mask(path, name)).astype(bool)
            self.assertTrue(self.np.array_equal(mask, expected), name)

    def test_instances_and_raters_of_one_label(self):
        first, second = self.np.zeros((4, 8, 8), dtype=bool), self.np.zeros((4, 8, 8), dtype=bool)
        first[0], second[2] = True, True
        path = self.tmp / "instances.seg.nrrd"
        masks = {Segment("LIVER_TUMOR", instance=1): first, Segment("LIVER_TUMOR", instance=2): second,
                 Segment("LIVER", rater="A"): first | second}
        self.assertEqual(write_segmentation(masks, self.reference, path), ["LIVER", "LIVER_TUMOR"])
        _, segments = read_segmentation(path)
        self.assertEqual(sorted(s.id for s in segments), ["LIVER@A", "LIVER_TUMOR#1", "LIVER_TUMOR#2"])
        both = self.sitk.GetArrayFromImage(read_label_mask(path, "LIVER_TUMOR")).astype(bool)
        self.assertTrue(self.np.array_equal(both, first | second))
        one = self.sitk.GetArrayFromImage(read_label_mask(path, "LIVER_TUMOR", instance=2)).astype(bool)
        self.assertTrue(self.np.array_equal(one, second))
        # a header knows each label once
        self.assertEqual(read_segmentation_labels(path), ["LIVER", "LIVER_TUMOR"])
        with self.assertRaises(ValueError):  # two segments of a label that nothing tells apart
            numbers = self.np.zeros((4, 8, 8), dtype=self.np.uint8)
            write_indexed_segmentation(numbers, [Segment("LIVER", 0, 1), Segment("LIVER", 0, 2)], self.reference, path)

    def test_more_than_255_segments_use_uint16(self):
        names = [label.name for label in LabelMap][:300]
        masks = {}
        for i, name in enumerate(names):
            mask = self.np.zeros((1, 20, 20), dtype=bool)
            mask.flat[i] = True
            masks[name] = mask
        path = self.tmp / "many.seg.nrrd"
        write_segmentation(masks, self.sitk.GetImageFromArray(self.np.zeros((1, 20, 20), dtype=self.np.uint8)), path)
        self.assertEqual(self.sitk.ReadImage(str(path)).GetPixelID(), self.sitk.sitkUInt16)
        self.assertEqual(read_segmentation_labels(path), names)

    def test_indexed_writer_skips_numbers_not_in_the_mask(self):
        numbers = self.np.zeros((2, 4, 4), dtype=self.np.uint8)
        numbers[0] = 2
        labels = [LabelMap.TIBIA_LEFT, LabelMap.FEMUR_LEFT]
        path = self.tmp / "indexed.seg.nrrd"
        written = write_indexed_segmentation(numbers, labels, self.sitk.GetImageFromArray(numbers), path)
        self.assertEqual(written, ["FEMUR_LEFT"])
        with self.assertRaises(ValueError):
            write_indexed_segmentation(numbers, labels * 2, self.sitk.GetImageFromArray(numbers), path)

    def test_wrong_suffix_is_rejected(self):
        with self.assertRaises(ValueError):
            write_segmentation(self.masks, self.reference, self.tmp / "001_000002.nii.gz")
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_a_name_that_is_no_label_is_rejected(self):
        with self.assertRaises(ValueError):
            write_segmentation({"THIGH_BONE": self.femur}, self.reference, self.tmp / "bad.seg.nrrd")

    def test_segment_added_by_hand_in_slicer(self):
        # no BoneHub tags: the label comes from the segment name
        image = self.sitk.GetImageFromArray(self.femur.astype(self.np.uint8))
        image.SetMetaData("Segment0_Name", "FEMUR_LEFT")
        image.SetMetaData("Segment0_LabelValue", "1")
        path = self.tmp / "manual.seg.nrrd"
        self.sitk.WriteImage(image, str(path))
        self.assertEqual(read_segmentation_labels(path), ["FEMUR_LEFT"])
        mask = self.sitk.GetArrayFromImage(read_label_mask(path, LabelMap.FEMUR_LEFT)).astype(bool)
        self.assertTrue(self.np.array_equal(mask, self.femur))


if __name__ == "__main__":
    unittest.main()
