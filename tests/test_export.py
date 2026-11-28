import shutil
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from bonehub_format import LabelMap as LM
from bonehub_format import read_label_mask, read_segmentation
from bonehub_format.export import Instance, compact_image, export_nii_segmentation, mapped_labels, read_image


class TestReadImage(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_direction_cosines_slightly_off_orthonormal(self):
        # as in TotalSegmentator: an sform rotated about z whose directions are off by rounding, and no qform
        angle = np.radians(18.35)
        affine = np.eye(4)
        affine[:3, :3] = np.diag([1.5, 1.5, 1.5]) @ np.array(
            [[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        affine[0, 1] += 3e-4  # no longer orthonormal
        affine[:3, 3] = [-251.3, 82.6, 93.1]
        data = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
        image = nib.Nifti1Image(data, affine)
        image.set_sform(affine, code=2)
        image.set_qform(None, code=0)
        path = self.tmp / "ct.nii.gz"
        nib.save(image, path)

        with self.assertRaises(RuntimeError):
            sitk.ReadImage(str(path))  # ITK refuses it
        result = read_image(path)
        self.assertEqual(result.GetSize(), (4, 5, 6))
        self.assertEqual(result.GetPixelID(), sitk.sitkInt16)
        np.testing.assert_allclose(result.GetSpacing(), np.linalg.norm(affine[:3, :3], axis=0))
        np.testing.assert_allclose(result.GetOrigin(), [251.3, -82.6, 93.1])  # RAS -> LPS
        direction = np.array(result.GetDirection()).reshape(3, 3)
        np.testing.assert_allclose(direction @ direction.T, np.eye(3), atol=1e-12)
        # voxel (i, j, k) of the NIfTI array is voxel (i, j, k) of the image
        self.assertEqual(result[3, 4, 5], data[3, 4, 5])
        far = np.array(result.TransformIndexToPhysicalPoint((3, 4, 5)))
        expected = np.diag([-1, -1, 1]) @ (affine @ [3, 4, 5, 1])[:3]
        self.assertLess(np.abs(far - expected).max(), 0.01)


class TestCompactImage(unittest.TestCase):
    def test_pixel_types(self):
        integral = sitk.GetImageFromArray(np.array([[[-1024.0, 3071.0]]]))  # float64 HU
        self.assertEqual(compact_image(integral).GetPixelID(), sitk.sitkInt16)
        fractional = sitk.GetImageFromArray(np.array([[[0.5, 1.25]]]))
        self.assertEqual(compact_image(fractional).GetPixelID(), sitk.sitkFloat32)
        wide = sitk.GetImageFromArray(np.array([[[0, 40000]]], dtype=np.int32))
        self.assertEqual(compact_image(wide).GetPixelID(), sitk.sitkInt32)
        mr = sitk.GetImageFromArray(np.array([[[0, 40000]]], dtype=np.uint16))
        self.assertEqual(compact_image(mr).GetPixelID(), sitk.sitkUInt16)
        nan = sitk.GetImageFromArray(np.array([[[np.nan, 1.0]]], dtype=np.float32))
        self.assertEqual(compact_image(nan).GetPixelID(), sitk.sitkFloat32)



class TestExportNiiSegmentation(unittest.TestCase):
    """Source label files to one BoneHub segmentation: wholes built from pieces, instances, layers."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, name: str, array: np.ndarray) -> Path:
        path = self.tmp / f"{name}.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(array.astype(np.uint8)), str(path))
        return path

    def region(self, path: Path, label, instance=None) -> np.ndarray:
        return sitk.GetArrayFromImage(read_label_mask(path, label, instance)).astype(bool)

    def test_a_whole_from_its_pieces_in_one_file(self):
        # KiTS: 1 kidney, 2 tumor, 3 cyst; the kidney is all three
        kits = np.zeros((4, 8, 8), dtype=np.uint8)
        kits[:, :4], kits[:, 4:6], kits[:, 6:] = 1, 2, 3
        mapping = {1: LM.KIDNEY, 2: [LM.KIDNEY, LM.KIDNEY_TUMOR], 3: [LM.KIDNEY, LM.KIDNEY_CYST]}
        out = self.tmp / "kits.seg.nrrd"
        self.assertEqual(export_nii_segmentation([self.write("kits", kits)], out, [mapping]),
                         ["KIDNEY", "KIDNEY_CYST", "KIDNEY_TUMOR"])
        self.assertTrue(self.region(out, "KIDNEY").all())
        self.assertTrue(np.array_equal(self.region(out, "KIDNEY_TUMOR"), kits == 2))
        layers, segments = read_segmentation(out)
        self.assertEqual(len(layers), 2)  # the pieces lie in the whole
        self.assertEqual(mapped_labels(mapping), ["KIDNEY", "KIDNEY_CYST", "KIDNEY_TUMOR"])

    def test_a_whole_from_pieces_in_several_files(self):
        # TotalSegmentator: one file per structure; the heart is heart.nii.gz and the left atrial appendage
        heart, appendage = np.zeros((4, 8, 8), dtype=np.uint8), np.zeros((4, 8, 8), dtype=np.uint8)
        heart[:, :4], appendage[:, 4:5] = 1, 1
        out = self.tmp / "ts.seg.nrrd"
        files = [self.write("heart", heart), self.write("atrial_appendage_left", appendage)]
        written = export_nii_segmentation(files, out, [{1: LM.HEART}, {1: [LM.HEART, LM.LEFT_ATRIAL_APPENDAGE]}])
        self.assertEqual(written, ["HEART", "LEFT_ATRIAL_APPENDAGE"])
        self.assertTrue(np.array_equal(self.region(out, "HEART"), (heart | appendage).astype(bool)))
        _, segments = read_segmentation(out)
        self.assertEqual([s.label.name for s in segments], ["HEART", "LEFT_ATRIAL_APPENDAGE"])  # one segment per label

    def test_a_whole_from_a_remainder_and_named_pieces_that_touch_other_labels(self):
        # MOOSE: body composition's skeletal muscle leaves out the named muscles, which touch its fat here and there
        composition, muscles = np.zeros((4, 8, 8), dtype=np.uint8), np.zeros((4, 8, 8), dtype=np.uint8)
        composition[:, :3], composition[:, 3:] = 1, 2  # skeletal muscle, subcutaneous fat
        muscles[:, 2:4] = 3  # a gluteus, over the muscle and the fat
        out = self.tmp / "moose.seg.nrrd"
        written = export_nii_segmentation(
            [self.write("composition", composition), self.write("muscles", muscles)], out,
            [{1: LM.MUSCLES, 2: LM.SUBCUTANEOUS_ADIPOSE_TISSUE}, {3: [LM.GLUTEUS_MAXIMUS_LEFT, LM.MUSCLES]}])
        self.assertEqual(written, ["MUSCLES", "GLUTEUS_MAXIMUS_LEFT", "SUBCUTANEOUS_ADIPOSE_TISSUE"])
        self.assertTrue(np.array_equal(self.region(out, "MUSCLES"), (composition == 1) | (muscles == 3)))
        self.assertTrue(np.array_equal(self.region(out, "SUBCUTANEOUS_ADIPOSE_TISSUE"), composition == 2))
        self.assertTrue(np.array_equal(self.region(out, "GLUTEUS_MAXIMUS_LEFT"), muscles == 3))

    def test_a_whole_and_its_piece_from_one_source_label(self):
        liver = np.zeros((2, 4, 4), dtype=np.uint8)
        liver[0], liver[1, :2] = 1, 2
        out = self.tmp / "liver.seg.nrrd"
        export_nii_segmentation([self.write("liver", liver)], out, [{1: LM.LIVER, 2: [LM.LIVER_TUMOR, LM.LIVER]}])
        self.assertTrue(np.array_equal(self.region(out, "LIVER"), liver > 0))
        self.assertTrue(np.array_equal(self.region(out, "LIVER_TUMOR"), liver == 2))

    def test_instances_and_overlapping_files(self):
        # PENGWIN: fragments of a bone are instances of its label; a second file overlaps the first
        fragments, body = np.zeros((4, 8, 8), dtype=np.uint8), np.zeros((4, 8, 8), dtype=np.uint8)
        fragments[:, :2], fragments[:, 2:4] = 11, 12
        body[:] = 1
        out = self.tmp / "pengwin.seg.nrrd"
        mapping = {11: Instance(LM.HIP_BONE_LEFT, 1), 12: Instance(LM.HIP_BONE_LEFT, 2)}
        written = export_nii_segmentation([self.write("frag", fragments), self.write("body", body)], out,
                                          [mapping, {1: LM.BODY}])
        self.assertEqual(written, ["HIP_BONE_LEFT", "BODY"])
        _, segments = read_segmentation(out)
        self.assertEqual(sorted((s.id, s.layer) for s in segments), [("BODY", 1), ("HIP_BONE_LEFT#1", 0), ("HIP_BONE_LEFT#2", 0)])
        self.assertTrue(np.array_equal(self.region(out, "HIP_BONE_LEFT", 2), fragments == 12))
        self.assertTrue(np.array_equal(self.region(out, "HIP_BONE_LEFT"), fragments > 0))
        self.assertTrue(self.region(out, "BODY").all())


if __name__ == "__main__":
    unittest.main()
