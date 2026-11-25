import shutil
import tempfile
import unittest
from pathlib import Path

import SimpleITK as sitk

from bonehub_format import ScanInfo
from bonehub_format.converter import ScanSource
from bonehub_format.selection import choose_scan, image_geometry, keep_one


class TestOneScanPerSubject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _scan(self, name: str, slices: int, thickness: float, days: int | None = None) -> ScanSource:
        image = sitk.Image([8, 8, slices], sitk.sitkInt16)
        image.SetSpacing((1.0, 1.0, thickness))
        path = self.tmp / f"{name}.nii.gz"
        sitk.WriteImage(image, str(path))
        return ScanSource(source_scan_id=name, info=ScanInfo(modality="CT", days_from_baseline=days), image_path=path)

    def test_geometry_is_the_extent_along_the_body_axis_and_the_slice_thickness(self):
        self.assertEqual(image_geometry(self._scan("geometry", 40, 2.5).image_path), (100.0, 2.5))

    def test_the_scan_with_the_most_data_is_kept(self):
        small, large = self._scan("small", 10, 1.0), self._scan("large", 200, 1.0)
        self.assertIs(choose_scan([small, large], data=lambda scan: scan is small), small)

    def test_then_the_largest_coverage_then_the_thinnest_slices_then_the_earliest(self):
        short, long = self._scan("short", 100, 1.0), self._scan("long", 100, 2.0)  # 100 mm and 200 mm
        self.assertIs(choose_scan([short, long]), long)
        thick, thin = self._scan("thick", 40, 5.0), self._scan("thin", 200, 1.0)  # both 200 mm
        self.assertIs(choose_scan([thick, thin]), thin)
        later, first = self._scan("later", 50, 1.0, days=30), self._scan("first", 50, 1.0, days=0)
        self.assertIs(choose_scan([later, first]), first)

    def test_keep_one_says_how_many_scans_were_left_out(self):
        scans = [self._scan(f"phase{i}", 50, 1.0) for i in range(3)]
        [kept] = keep_one(scans, what="phases")
        self.assertIs(kept, scans[0])
        self.assertIn("2 other phases of this subject", kept.info.remarks)
        [alone] = keep_one([self._scan("alone", 50, 1.0)])
        self.assertIsNone(alone.info.remarks)


if __name__ == "__main__":
    unittest.main()
