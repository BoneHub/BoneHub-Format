import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import support
from bonehub_format import LabelMap as LM
from bonehub_format import Database, layout, read_label_mask, read_segmentation, update_model_files, update_segmentation, validate


class TestEditInPlace(unittest.TestCase):
    """Changing the labels of a converted scan, keeping the JSON files in step with the data files."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "root"
        source = support.make_source(self.tmp / "source", {"p1": ["ct"]})
        support.SyntheticConverter(source).export_to_bonehub_format(self.root, 1, num_workers=1, verbose=False)
        self.scan = Database(self.root).dataset(1).subject(1).scan(1)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def mask(self, boxes: dict) -> dict[str, sitk.Image]:
        """label -> its region, one image per label on the scan's grid"""
        reference = sitk.ReadImage(str(self.scan.image_path))
        masks = {}
        for label, box in boxes.items():
            array = np.zeros(support.SHAPE, dtype=np.uint8)
            array[box] = 1
            image = sitk.GetImageFromArray(array)
            image.CopyInformation(reference)
            masks[label] = image
        return masks

    def voxels(self, label, number: int = 1) -> int:
        path = layout.data_file(self.scan.path, "segmentation", number)
        return int(sitk.GetArrayFromImage(read_label_mask(path, label)).sum())

    def layers(self, number: int = 1) -> dict[str, int]:
        _, segments = read_segmentation(layout.data_file(self.scan.path, "segmentation", number))
        return {segment.label.name: segment.layer for segment in segments}

    def origins(self, kind: str, number: int = 1) -> dict | None:
        path = layout.data_json(self.scan.path, kind, number)
        return json.loads(path.read_text())["labels"] if path.is_file() else None

    def test_add_a_label(self):
        mask = self.mask({LM.PATELLA_LEFT: np.s_[9:11, 0:3, 0:3]})
        labels = update_segmentation(self.scan, mask, "model:Seg@1.0", number=1)
        self.assertEqual(labels, ["FEMUR_LEFT", "FEMUR_RIGHT", "PATELLA_LEFT"])
        self.assertEqual(
            self.origins("segmentation"),
            {"FEMUR_LEFT": "source", "FEMUR_RIGHT": "source", "PATELLA_LEFT": "model:Seg@1.0"},
        )
        self.assertEqual(self.voxels(LM.PATELLA_LEFT), 2 * 3 * 3)
        self.assertEqual(validate(self.root), [])

    def test_replace_a_label_and_overlap(self):
        femur_right = np.s_[5:9, 6:10, 9:13]  # where the source has FEMUR_RIGHT
        box = np.s_[0:2, 0:2, 0:2]
        update_segmentation(self.scan, self.mask({LM.FEMUR_LEFT: box}), "manual", number=1)
        self.assertEqual(self.voxels(LM.FEMUR_LEFT), 8)  # the old FEMUR_LEFT is gone
        self.assertEqual(self.origins("segmentation")["FEMUR_LEFT"], "manual")

        update_segmentation(self.scan, self.mask({LM.TIBIA_RIGHT: femur_right}), "manual", overlap="keep", number=1)
        self.assertNotIn("TIBIA_RIGHT", self.scan.labels("segmentation"))  # every voxel was FEMUR_RIGHT's
        update_segmentation(self.scan, self.mask({LM.TIBIA_RIGHT: femur_right}), "manual", overlap="replace", number=1)
        self.assertEqual(self.scan.labels("segmentation"), ["FEMUR_LEFT", "TIBIA_RIGHT"])  # replaced
        self.assertEqual(self.layers(), {"FEMUR_LEFT": 0, "TIBIA_RIGHT": 0})
        self.assertEqual(validate(self.root), [])

    def test_an_overlapping_label_goes_into_a_layer_of_its_own(self):
        cortical = "FEMUR_PROXIMAL_CORTICAL_RIGHT"
        update_segmentation(self.scan, self.mask({cortical: np.s_[5:6, 6:10, 9:13]}), "manual", number=1)
        self.assertEqual(self.layers(), {"FEMUR_LEFT": 0, "FEMUR_RIGHT": 0, cortical: 1})
        self.assertEqual((self.voxels(LM.FEMUR_RIGHT), self.voxels(cortical)), (4 * 4 * 4, 4 * 4))  # both whole
        fields = json.loads(layout.data_json(self.scan.path, "segmentation", 1).read_text())
        self.assertEqual(fields["labels"][cortical], "manual")
        self.assertEqual(validate(self.root), [])

        # removing the label that overlapped leaves one layer
        update_segmentation(self.scan, remove=["FEMUR_RIGHT"], number=1)
        self.assertEqual(self.layers(), {"FEMUR_LEFT": 0, cortical: 0})
        self.assertEqual(self.voxels(cortical), 4 * 4)
        self.assertEqual(validate(self.root), [])

        # a segmentation file is added label by label: here its copy in a new file of the scan
        copy = self.tmp / "copy.seg.nrrd"
        shutil.copy(layout.data_file(self.scan.path, "segmentation", 1), copy)
        self.assertEqual(update_segmentation(self.scan, copy, "manual"), ["FEMUR_LEFT", cortical])
        self.assertEqual(self.voxels(cortical, number=2), 4 * 4)
        self.assertEqual(validate(self.root), [])

    def test_remove_labels(self):
        self.assertEqual(update_segmentation(self.scan, remove=["FEMUR_LEFT"], number=1), ["FEMUR_RIGHT"])
        self.assertEqual(update_segmentation(self.scan, remove=["FEMUR_RIGHT"], number=1), [])
        self.assertEqual(self.scan.files("segmentation"), [])
        self.assertFalse(layout.data_file(self.scan.path, "segmentation", 1).exists())
        self.assertIsNone(self.origins("segmentation"))
        self.assertEqual(validate(self.root), [])

    def test_bad_input_changes_nothing(self):
        before = layout.data_file(self.scan.path, "segmentation", 1).read_bytes()
        shifted = self.mask({LM.PATELLA_LEFT: np.s_[0:1, 0:1, 0:1]})
        shifted[LM.PATELLA_LEFT].SetOrigin((0.0, 0.0, 0.0))
        cases = [
            (shifted, "model:Seg@1.0", 1),
            (self.mask({LM.PATELLA_LEFT: np.s_[0:1, 0:1, 0:1]}), None, 1),  # no origin
            (self.mask({LM.PATELLA_LEFT: np.s_[0:1, 0:1, 0:1]}), "by hand", 1),  # not an origin
            (self.mask({"THIGH_BONE": np.s_[0:1, 0:1, 0:1]}), "manual", 1),  # not a label
            (self.mask({LM.PATELLA_LEFT: np.s_[0:1, 0:1, 0:1]}), "manual", 2),  # no segmentation-02
        ]
        for mask, origin, number in cases:
            with self.subTest(origin=origin, number=number), self.assertRaises(ValueError):
                update_segmentation(self.scan, mask, origin, number=number)
        self.assertEqual(layout.data_file(self.scan.path, "segmentation", 1).read_bytes(), before)
        self.assertEqual(validate(self.root), [])

    def test_model_files(self):
        iges = self.tmp / "femur.IGES"
        iges.write_bytes(b"IGES")
        self.assertEqual(update_model_files(self.scan, "nurbs", {"FEMUR_LEFT": iges}, origin="derived:tool@1.0"), ["FEMUR_LEFT"])
        self.assertEqual(self.origins("nurbs"), {"FEMUR_LEFT": "derived:tool@1.0"})
        self.assertTrue((self.scan.path / "nurbs-01" / "FEMUR_LEFT.iges").is_file())
        self.assertTrue(layout.model_zip(self.scan.path, "nurbs", 1).is_file())
        self.assertEqual(validate(self.root, publishing=False), [])

        # a downloaded copy has only the zips
        shutil.rmtree(self.scan.path / "mesh-01")
        stl = self.tmp / "tibia.stl"
        stl.write_bytes(b"solid TIBIA_LEFT\nendsolid TIBIA_LEFT\n")
        update_model_files(self.scan, "mesh", {"TIBIA_LEFT": stl}, origin="manual", number=1)
        self.assertEqual(self.origins("mesh"), {"FEMUR_LEFT": "derived:test-mesher@1.0", "TIBIA_LEFT": "manual"})

        self.assertEqual(update_model_files(self.scan, "nurbs", remove=["FEMUR_LEFT"], number=1), [])
        self.assertFalse(layout.model_zip(self.scan.path, "nurbs", 1).exists())
        self.assertIsNone(self.origins("nurbs"))
        with self.assertRaises(ValueError):
            update_model_files(self.scan, "mesh", {"TIBIA_LEFT": iges}, origin="manual", number=1)  # .iges is not a mesh
        self.assertEqual(validate(self.root), [])

    def test_a_new_file_of_another_source_and_numbering_after_a_removal(self):
        mask = self.mask({LM.PATELLA_LEFT: np.s_[9:11, 0:3, 0:3]})
        self.assertEqual(update_segmentation(self.scan, mask, "manual", source=support.SOURCE_ID), ["PATELLA_LEFT"])
        second = json.loads(layout.data_json(self.scan.path, "segmentation", 2).read_text())
        self.assertEqual(second["source"], support.SOURCE_ID)
        self.assertEqual(second["source_scan_id"], "ds-0001/sub-000001/scan-01")  # a new file: the scan's path
        self.assertEqual(self.origins("segmentation"), {"FEMUR_LEFT": "source", "FEMUR_RIGHT": "source"})  # untouched
        self.assertEqual(validate(self.root), [])

        update_segmentation(self.scan, remove=["FEMUR_LEFT", "FEMUR_RIGHT"], number=1)  # segmentation-02 becomes -1
        self.assertEqual(self.origins("segmentation"), {"PATELLA_LEFT": "manual"})
        self.assertFalse(layout.data_file(self.scan.path, "segmentation", 2).exists())
        self.assertEqual(validate(self.root), [])

        with self.assertRaises(ValueError):  # an existing file keeps its source
            update_segmentation(self.scan, mask, "manual", number=1, source=99)


if __name__ == "__main__":
    unittest.main()
