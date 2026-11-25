import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import support
from bonehub_format import LabelMap as LM
from bonehub_format import Database, layout, read_segmentation, update_model_files, update_segmentation, validate


class TestEditInPlace(unittest.TestCase):
    """Changing the labels of a converted scan, keeping scan.json in step with the files."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "root"
        source = support.make_source(self.tmp / "source", {"p1": ["ct"]})
        support.SyntheticConverter(source).export_to_bonehub_format(self.root, 1, num_workers=1, verbose=False)
        self.scan = Database(self.root).dataset(1).subject(1).scan(1)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def mask(self, boxes: dict[int, tuple[slice, ...]]) -> sitk.Image:
        array = np.zeros(support.SHAPE, dtype=np.int32)
        for value, box in boxes.items():
            array[box] = value
        image = sitk.GetImageFromArray(array)
        image.CopyInformation(sitk.ReadImage(str(self.scan.image_path)))
        return image

    def segmentation(self) -> np.ndarray:
        return sitk.GetArrayFromImage(read_segmentation(self.scan.segmentation_path(1)))

    def origins(self, kind: str, dataset_id: int = 1) -> dict | None:
        provider = self.scan.info.provider(dataset_id)
        return getattr(provider, kind) if provider else None

    def test_add_a_label(self):
        labels = update_segmentation(self.scan, self.mask({LM.PATELLA_LEFT.value: np.s_[9:11, 0:3, 0:3]}), "model:Seg@1.0")
        self.assertEqual(labels, ["FEMUR_LEFT", "FEMUR_RIGHT", "PATELLA_LEFT"])
        self.assertEqual(
            self.origins("segmentation"),
            {"FEMUR_LEFT": "source", "FEMUR_RIGHT": "source", "PATELLA_LEFT": "model:Seg@1.0"},
        )
        self.assertEqual(int((self.segmentation() == LM.PATELLA_LEFT.value).sum()), 2 * 3 * 3)
        self.assertEqual(validate(self.root), [])

    def test_replace_a_label_and_overlap(self):
        femur_right = np.s_[5:9, 6:10, 9:13]  # where the source has FEMUR_RIGHT
        box = np.s_[0:2, 0:2, 0:2]
        update_segmentation(self.scan, self.mask({LM.FEMUR_LEFT.value: box}), "manual")
        segmentation = self.segmentation()
        self.assertEqual(int((segmentation == LM.FEMUR_LEFT.value).sum()), 8)  # the old FEMUR_LEFT is gone
        self.assertEqual(self.origins("segmentation")["FEMUR_LEFT"], "manual")

        update_segmentation(self.scan, self.mask({LM.TIBIA_RIGHT.value: femur_right}), "manual", overlap="keep")
        self.assertNotIn("TIBIA_RIGHT", self.scan.info.labels("segmentation"))  # every voxel was FEMUR_RIGHT's
        update_segmentation(self.scan, self.mask({LM.TIBIA_RIGHT.value: femur_right}), "manual")
        self.assertEqual(self.scan.info.labels("segmentation"), ["FEMUR_LEFT", "TIBIA_RIGHT"])  # replaced
        self.assertEqual(validate(self.root), [])

    def test_remove_labels(self):
        self.assertEqual(update_segmentation(self.scan, remove=["FEMUR_LEFT"]), ["FEMUR_RIGHT"])
        self.assertEqual(update_segmentation(self.scan, remove=["FEMUR_RIGHT"]), [])
        self.assertIsNone(self.scan.segmentation_path(1))
        self.assertIsNone(self.origins("segmentation"))
        self.assertEqual(validate(self.root), [])

    def test_bad_input_changes_nothing(self):
        before = self.scan.segmentation_path(1).read_bytes()
        shifted = self.mask({LM.PATELLA_LEFT.value: np.s_[0:1, 0:1, 0:1]})
        shifted.SetOrigin((0.0, 0.0, 0.0))
        cases = [
            (shifted, "model:Seg@1.0"),
            (self.mask({LM.PATELLA_LEFT.value: np.s_[0:1, 0:1, 0:1]}), None),  # no origin
            (self.mask({LM.PATELLA_LEFT.value: np.s_[0:1, 0:1, 0:1]}), "by hand"),  # not an origin
            (self.mask({12345: np.s_[0:1, 0:1, 0:1]}), "manual"),  # not a label
        ]
        for mask, origin in cases:
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                update_segmentation(self.scan, mask, origin)
        self.assertEqual(self.scan.segmentation_path(1).read_bytes(), before)
        self.assertEqual(validate(self.root), [])

    def test_model_files(self):
        iges = self.tmp / "femur.IGES"
        iges.write_bytes(b"IGES")
        self.assertEqual(update_model_files(self.scan, "nurbs", {"FEMUR_LEFT": iges}, origin="derived:tool@1.0"), ["FEMUR_LEFT"])
        self.assertEqual(self.origins("nurbs"), {"FEMUR_LEFT": "derived:tool@1.0"})
        self.assertTrue((self.scan.path / "nurbs.ds-0001" / "FEMUR_LEFT.iges").is_file())
        self.assertTrue(layout.model_zip(self.scan.path, "nurbs", 1).is_file())
        self.assertEqual(validate(self.root, publishing=False), [])

        # a downloaded copy has only the zips
        shutil.rmtree(self.scan.path / "mesh.ds-0001")
        stl = self.tmp / "tibia.stl"
        stl.write_bytes(b"solid TIBIA_LEFT\nendsolid TIBIA_LEFT\n")
        update_model_files(self.scan, "mesh", {"TIBIA_LEFT": stl}, origin="manual")
        self.assertEqual(self.origins("mesh"), {"FEMUR_LEFT": "derived:test-mesher@1.0", "TIBIA_LEFT": "manual"})

        self.assertEqual(update_model_files(self.scan, "nurbs", remove=["FEMUR_LEFT"]), [])
        self.assertFalse(layout.model_zip(self.scan.path, "nurbs", 1).exists())
        self.assertIsNone(self.origins("nurbs"))
        with self.assertRaises(ValueError):
            update_model_files(self.scan, "mesh", {"TIBIA_LEFT": iges}, origin="manual")  # .iges is not a mesh
        self.assertEqual(validate(self.root), [])

    def test_edit_the_files_of_another_provider(self):
        mask = self.mask({LM.PATELLA_LEFT.value: np.s_[9:11, 0:3, 0:3]})
        self.assertEqual(update_segmentation(self.scan, mask, "manual", dataset_id=5), ["PATELLA_LEFT"])
        self.assertTrue((self.scan.path / "segmentation.ds-0005.seg.nrrd").is_file())
        provider = self.scan.info.provider(5)
        self.assertEqual(provider.source_scan_id, "ds-0001/sub-000001/scan-01")  # a new provider: the scan's path
        self.assertEqual(provider.segmentation, {"PATELLA_LEFT": "manual"})
        self.assertEqual(self.origins("segmentation"), {"FEMUR_LEFT": "source", "FEMUR_RIGHT": "source"})  # untouched


if __name__ == "__main__":
    unittest.main()
