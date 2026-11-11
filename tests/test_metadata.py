import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from bonehub_format import (
    LICENSES,
    RESTRICTED,
    SCHEMA_VERSION,
    DatasetInfo,
    Root,
    ScanInfo,
    SubjectInfo,
    is_compatible_schema_version,
    layout,
)


class TestDatasetInfo(unittest.TestCase):
    def test_license(self):
        self.assertEqual(DatasetInfo(name="A", license="CC-BY-NC-SA-4.0").license, "CC-BY-NC-SA-4.0")
        self.assertEqual(DatasetInfo(name="A", license=RESTRICTED).license, "restricted")
        for bad in ("CC BY 4.0", "cc-by-4.0", "Other", ""):
            with self.subTest(license=bad), self.assertRaises(ValidationError):
                DatasetInfo(name="A", license=bad)
        self.assertIn("CC-BY-NC-3.0", LICENSES)

    def test_annotated_labels_are_checked_and_sorted(self):
        info = DatasetInfo(name="A", annotated_labels=["FEMUR_RIGHT", "SKULL", "FEMUR_LEFT", "SKULL"])
        self.assertEqual(info.annotated_labels, ["SKULL", "FEMUR_LEFT", "FEMUR_RIGHT"])
        with self.assertRaises(ValidationError):
            DatasetInfo(name="A", annotated_labels=["NOT_A_BONE"])
        self.assertEqual(DatasetInfo(name="A", annotated_labels=[]).annotated_labels, [])

    def test_name_is_required_and_unknown_fields_are_refused(self):
        with self.assertRaises(ValidationError):
            DatasetInfo()
        with self.assertRaises(ValidationError):
            DatasetInfo(name="A", modality="CT")  # modality belongs to the scan


class TestSubjectInfo(unittest.TestCase):
    def test_gender(self):
        for value in ("M", "F", "O", None):
            self.assertEqual(SubjectInfo(source_subject_id="1", gender=value).gender, value)
        for bad in ("", "male", "m"):
            with self.subTest(gender=bad), self.assertRaises(ValidationError):
                SubjectInfo(source_subject_id="1", gender=bad)


class TestScanInfo(unittest.TestCase):
    def test_origins(self):
        good = ["source", "manual", "model:TotalSegmentator@2.4.0", "derived:mesh2nurbs-rhino3d@1.0"]
        for origin in good:
            ScanInfo(source_scan_id="a", mesh={"FEMUR_LEFT": origin})
        for bad in ("Source", "model:TotalSegmentator", "derived:@1.0", "model:a b@1", 1):
            with self.subTest(origin=bad), self.assertRaises(ValidationError):
                ScanInfo(source_scan_id="a", segmentation={"FEMUR_LEFT": bad})

    def test_labels_are_checked_and_kept_in_label_map_order(self):
        scan = ScanInfo(source_scan_id="a")
        scan.set_labels("segmentation", ["TIBIA_LEFT", "FEMUR_LEFT"], "source")
        self.assertEqual(scan.labels("segmentation"), ["FEMUR_LEFT", "TIBIA_LEFT"])
        scan.set_labels("nurbs", ["FEMUR_LEFT"], {"FEMUR_LEFT": "manual"})
        self.assertEqual(scan.nurbs, {"FEMUR_LEFT": "manual"})
        scan.set_labels("mesh", [], "source")
        self.assertIsNone(scan.mesh)
        with self.assertRaises(ValueError):
            scan.set_labels("segmentation", ["NOT_A_BONE"], "source")
        with self.assertRaises(ValueError):
            scan.set_labels("mesh", ["FEMUR_LEFT"], {})  # no origin for the label
        with self.assertRaises(ValueError):
            scan.labels("landmarks")

    def test_other_fields(self):
        scan = ScanInfo(source_scan_id="a", weight=81, registered_to="scan-02", postmortem=True, license="CC-BY-NC-3.0")
        self.assertEqual(scan.to_dict(), {"source_scan_id": "a", "weight": 81.0, "postmortem": True, "registered_to": "scan-02", "license": "CC-BY-NC-3.0"})
        for field, bad in (("registered_to", "2"), ("age", -1), ("weight", 0), ("postmortem", "yes"), ("license", "MIT-ish")):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ScanInfo(source_scan_id="a", **{field: bad})


class TestSchemaVersion(unittest.TestCase):
    def test_compatibility(self):
        major, minor = SCHEMA_VERSION.split(".")
        self.assertTrue(is_compatible_schema_version(SCHEMA_VERSION))
        self.assertFalse(is_compatible_schema_version(f"{major}.{int(minor) + 1}"))  # written by a newer package
        self.assertFalse(is_compatible_schema_version(f"{int(major) + 1}.0"))
        for old in (None, "", "0.3.0", "x.y", "1"):
            self.assertFalse(is_compatible_schema_version(old))

    def test_reader_refuses_another_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / layout.dataset_dir_name(1)
            folder.mkdir()
            (folder / layout.DATASET_JSON).write_text(json.dumps({"name": "A", "schema_version": "0.3.0"}))
            with self.assertRaises(ValueError):
                Root(tmp).dataset(1).info


class TestLayout(unittest.TestCase):
    def test_names(self):
        self.assertEqual(layout.scan_key(1, 23, 4), "ds-001/sub-000023/scan-04")
        self.assertEqual(layout.parse_scan_key("ds-001/sub-000023/scan-04"), (1, 23, 4))
        self.assertEqual(layout.parse_subject_dir("sub-000023"), 23)
        for bad in ("sub-23", "sub-000000", "scan-01"):
            self.assertIsNone(layout.parse_subject_dir(bad))
        with self.assertRaises(ValueError):
            layout.subject_dir_name(10_000)  # a folder on Hugging Face holds at most 10,000 entries
        self.assertEqual(layout.model_zip(Path("x"), "nurbs"), Path("x") / "nurbs.zip")


if __name__ == "__main__":
    unittest.main()
