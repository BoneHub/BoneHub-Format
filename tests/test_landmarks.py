import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import support
from bonehub_format import Database, LabelMap, geometry_of, is_landmark, landmark_on, pack, unpack, validate
from bonehub_format.landmarks import problems, read_landmarks, write_landmarks


def document(*entries):
    return {"coordinate_system": "LPS", "units": "mm", "landmarks": list(entries)}


class TestTheFormat(unittest.TestCase):
    def test_landmark_labels(self):
        self.assertTrue(is_landmark(LabelMap.LANDMARK_FEMUR_HEAD_CENTER_LEFT))
        self.assertFalse(is_landmark(LabelMap.FEMUR_LEFT))
        self.assertEqual(landmark_on(LabelMap.LANDMARK_FEMUR_HEAD_CENTER_LEFT), LabelMap.FEMUR_LEFT)
        self.assertEqual(landmark_on(LabelMap.LANDMARK_SACRUM_PROMONTORY), LabelMap.SACRUM)
        self.assertEqual(landmark_on(LabelMap.LANDMARK_ADDUCTOR_MAGNUS_DISTAL_ORIGIN_RIGHT), LabelMap.ADDUCTOR_MAGNUS_DISTAL_RIGHT)
        self.assertEqual(geometry_of(LabelMap.LANDMARK_FEMUR_HEAD_CENTER), ("point", "sphere"))
        self.assertIn("area", geometry_of(LabelMap.LANDMARK_MULTIFIDUS_ORIGIN_LEFT))

    def test_valid_entries(self):
        self.assertEqual(problems(document(*support.LANDMARKS), {"LANDMARK_GLUTEUS_MEDIUS_ORIGIN_LEFT.stl"}), [])
        circle = {"name": "LANDMARK_SCAPULA_GLENOID_CENTER_LEFT", "type": "circle", "center": [0, 0, 0], "radius": 12.4, "normal": None}
        frame = {"name": "LANDMARK_LUNATE_PRINCIPAL_AXES_RIGHT", "type": "frame", "origin": [1, 2, 3],
                 "axes": [[1, 0, 0], [0, 0, 1], [0, -1, 0]], "values": {"moments": [1.0, 2.0, 3.0]}}
        cylinder = {"name": "LANDMARK_HUMERUS_SHAFT_AXIS_LEFT", "type": "cylinder", "points": [[0, 0, 0], [0, 0, 50]], "radius": 9.5}
        self.assertEqual(problems(document(circle, frame, cylinder)), [])

    def test_mistakes_are_caught(self):
        cases = [
            ({"name": "FEMUR_LEFT", "type": "point", "position": [0, 0, 0]}, "not a landmark label"),
            ({"name": "LANDMARK_FEMUR_GREATER_TROCHANTER_LEFT", "type": "sphere", "center": [0, 0, 0], "radius": 2}, "is a sphere"),
            ({"name": "LANDMARK_FEMUR_GREATER_TROCHANTER_LEFT", "type": "point", "position": [0, 0]}, "is not [x, y, z]"),
            ({"name": "LANDMARK_FEMUR_HEAD_CENTER_LEFT", "type": "sphere", "center": [0, 0, 0], "radius": -1}, "positive"),
            ({"name": "LANDMARK_RADIUS_COORDINATE_SYSTEM_LEFT", "type": "frame", "origin": [0, 0, 0],
              "axes": [[1, 0, 0], [1, 0, 0], [0, 0, 1]]}, "orthonormal"),
            ({"name": "LANDMARK_FEMUR_GREATER_TROCHANTER_LEFT", "type": "point", "position": [0, 0, 0], "rater": 0}, "positive integer"),
            ({"name": "LANDMARK_FEMUR_GREATER_TROCHANTER_LEFT", "type": "point", "position": [0, 0, 0], "colour": "red"}, "unknown fields"),
            ({"name": "LANDMARK_GLUTEUS_MEDIUS_ORIGIN_LEFT", "type": "area", "file": "areas.nrrd"}, "names a file the folder does not hold"),
            ({"name": "LANDMARK_GLUTEUS_MEDIUS_ORIGIN_LEFT", "type": "area", "file": "LANDMARK_GLUTEUS_MINIMUS_ORIGIN_LEFT.stl"}, "another landmark"),
        ]
        for entry, expected in cases:
            found = problems(document(entry), {"LANDMARK_GLUTEUS_MINIMUS_ORIGIN_LEFT.stl"})
            self.assertTrue(any(expected in p for p in found), (entry, found))
        self.assertTrue(problems({"coordinate_system": "RAS", "units": "mm", "landmarks": []}))

    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "landmarks-01"
            with self.assertRaises(ValueError):  # the area's STL is not there yet
                write_landmarks(folder, support.LANDMARKS)
            (folder / "LANDMARK_GLUTEUS_MEDIUS_ORIGIN_LEFT.stl").write_bytes(b"solid x\nendsolid x\n")
            write_landmarks(folder, support.LANDMARKS)
            self.assertEqual(read_landmarks(folder)["landmarks"], support.LANDMARKS)


class TestAConvertedDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        source = support.make_source(cls.tmp / "source", {"case_1": ["ct"]})
        cls.root = cls.tmp / "root"
        support.SyntheticConverter(source, landmarks=True).export_to_bonehub_format(cls.root, 1, num_workers=1, verbose=False)
        cls.scan = cls.root / "ds-0001" / "sub-000001" / "scan-01"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_files(self):
        info = json.loads((self.scan / "landmarks-01.json").read_text())
        self.assertEqual(info["labels"], {
            "LANDMARK_FEMUR_HEAD_CENTER_LEFT": "manual", "LANDMARK_FEMUR_GREATER_TROCHANTER_LEFT": "manual",
            "LANDMARK_KNEE_JOINT_FLEXION_AXIS_LEFT": "manual", "LANDMARK_GLUTEUS_MEDIUS_ORIGIN_LEFT": "manual"})
        with zipfile.ZipFile(self.scan / "landmarks-01.zip") as archive:
            self.assertEqual(sorted(archive.namelist()), ["LANDMARK_GLUTEUS_MEDIUS_ORIGIN_LEFT.stl", "landmarks.json"])

    def test_reader_and_validation(self):
        scan = Database(self.root).dataset(1).subject(1).scan(1)
        [landmarks] = scan.files("landmarks")
        self.assertEqual(landmarks.landmarks(), support.LANDMARKS)
        self.assertIn("LANDMARK_FEMUR_HEAD_CENTER_LEFT", scan.labels("landmarks"))
        self.assertEqual(validate(self.root), [])
        shutil.rmtree(self.scan / "landmarks-01")  # as downloaded: the zip only
        self.assertEqual(landmarks.landmarks(), support.LANDMARKS)
        self.assertEqual(validate(self.root), [])
        unpack(self.root)
        self.assertTrue((self.scan / "landmarks-01" / "landmarks.json").is_file())
        self.assertEqual(pack(self.root), [])

    def test_a_landmark_in_a_segmentation_is_reported(self):
        path = self.scan / "landmarks-01.json"
        original = path.read_text()
        try:
            info = json.loads(original)
            info["labels"]["FEMUR_LEFT"] = "manual"
            path.write_text(json.dumps(info))
            self.assertTrue(any("are not landmarks" in p for p in validate(self.root)))
        finally:
            path.write_text(original)


if __name__ == "__main__":
    unittest.main()
