import shutil
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from bonehub_format.export import compact_image, read_image


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


if __name__ == "__main__":
    unittest.main()
