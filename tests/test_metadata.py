import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from bonehub_format import (
    LICENSES,
    RESTRICTED,
    SCHEMA_VERSION,
    Database,
    DatasetInfo,
    FileInfo,
    LabeledFileInfo,
    Licenses,
    ScanInfo,
    SourceInfo,
    SubjectInfo,
    SubsetInfo,
    file_license,
    is_compatible_schema_version,
    layout,
)


class TestSourceInfo(unittest.TestCase):
    def test_license_per_kind(self):
        info = SourceInfo(name="A", license=Licenses(image=RESTRICTED, segmentation="CC-BY-4.0"))
        self.assertEqual((info.license_of("image"), info.license_of("segmentation"), info.license_of("mesh")), ("restricted", "CC-BY-4.0", None))
        self.assertEqual(Licenses.all("MIT").to_dict(), {"image": "MIT", "segmentation": "MIT", "mesh": "MIT", "nurbs": "MIT"})
        for bad in ("CC BY 4.0", "cc-by-4.0", "Other", ""):
            with self.subTest(license=bad), self.assertRaises(ValidationError):
                Licenses(image=bad)
        with self.assertRaises(ValidationError):
            SourceInfo(name="A", license="CC-BY-4.0")  # a license is given per kind
        self.assertIn("CC-BY-NC-3.0", LICENSES)

    def test_subset_license_overrides_the_source(self):
        info = SourceInfo(
            name="CADS",
            license=Licenses.all("CC-BY-4.0"),
            subsets={"0004_lits": SubsetInfo(name="LiTS", license=Licenses(image="CC-BY-NC-SA-4.0"))},
        )
        self.assertEqual(info.license_of("image", "0004_lits"), "CC-BY-NC-SA-4.0")
        self.assertEqual(info.license_of("segmentation", "0004_lits"), "CC-BY-4.0")  # the subset gives none
        self.assertEqual(info.license_of("image", "unknown"), "CC-BY-4.0")

    def test_annotated_labels_are_checked_and_sorted(self):
        info = SourceInfo(name="A", annotated_labels=["FEMUR_RIGHT", "SKULL", "FEMUR_LEFT", "SKULL"])
        self.assertEqual(info.annotated_labels, ["SKULL", "FEMUR_LEFT", "FEMUR_RIGHT"])
        with self.assertRaises(ValidationError):
            SourceInfo(name="A", annotated_labels=["NOT_A_BONE"])
        self.assertEqual(SourceInfo(name="A", annotated_labels=[]).annotated_labels, [])


class TestDatasetInfo(unittest.TestCase):
    def test_sources_are_keyed_by_bonehub_id(self):
        info = DatasetInfo(name="Visible Human", sources={153: SourceInfo(name="CT"), 109: SourceInfo(name="Cryosections")})
        self.assertEqual(list(info.sources), [109, 153])
        self.assertEqual(list(info.to_dict()["sources"]), ["109", "153"])  # JSON keys are strings
        self.assertEqual(DatasetInfo.model_validate(info.to_dict()), info)  # and read back as BoneHub IDs
        self.assertEqual(info.source(153).name, "CT")
        self.assertIsNone(info.source(27))
        for bad in ({}, {"ds-0153": {"name": "A"}}, {"0": {"name": "A"}}):
            with self.subTest(sources=bad), self.assertRaises(ValidationError):
                DatasetInfo(name="A", sources=bad)

    def test_name_and_sources_are_required_and_unknown_fields_are_refused(self):
        with self.assertRaises(ValidationError):
            DatasetInfo(sources={1: SourceInfo(name="A")})
        with self.assertRaises(ValidationError):
            DatasetInfo(name="A")
        with self.assertRaises(ValidationError):
            DatasetInfo(name="A", sources={1: SourceInfo(name="A")}, modality="CT")  # modality belongs to the scan
        with self.assertRaises(ValidationError):
            DatasetInfo(name="A", sources={1: SourceInfo(name="A")}, derived_from=[2])  # sources replace it


class TestSubjectInfo(unittest.TestCase):
    def test_gender(self):
        for value in ("M", "F", "O", None):
            self.assertEqual(SubjectInfo(sources={1: "1"}, gender=value).gender, value)
        for bad in ("", "male", "m"):
            with self.subTest(gender=bad), self.assertRaises(ValidationError):
                SubjectInfo(sources={1: "1"}, gender=bad)

    def test_sources_are_keyed_and_sorted_by_bonehub_id(self):
        info = SubjectInfo.model_validate({"sources": {"134": "0001", "40": "PETCT_1"}})
        self.assertEqual(list(info.sources), [40, 134])
        self.assertEqual(info.source_subject_id(134), "0001")
        self.assertIsNone(info.source_subject_id(18))
        for bad in ("ds-0040", "-1", "0", "x"):
            with self.subTest(key=bad), self.assertRaises(ValidationError):
                SubjectInfo(sources={bad: "1"})


class TestFileInfo(unittest.TestCase):
    def test_origins(self):
        good = ["source", "manual", "model:TotalSegmentator@2.4.0", "derived:mesh2nurbs-rhino3d@1.0"]
        for origin in good:
            LabeledFileInfo(source=1, source_scan_id="a", labels={"FEMUR_LEFT": origin})
        for bad in ("Source", "model:TotalSegmentator", "derived:@1.0", "model:a b@1", 1):
            with self.subTest(origin=bad), self.assertRaises(ValidationError):
                LabeledFileInfo(source=1, source_scan_id="a", labels={"FEMUR_LEFT": bad})

    def test_labels_are_checked_and_kept_in_label_map_order(self):
        info = LabeledFileInfo(source=1, source_scan_id="a")
        info.set_labels(["TIBIA_LEFT", "FEMUR_LEFT"], "source")
        self.assertEqual(list(info.labels), ["FEMUR_LEFT", "TIBIA_LEFT"])
        info.set_labels(["FEMUR_LEFT"], {"FEMUR_LEFT": "manual"})
        self.assertEqual(info.labels, {"FEMUR_LEFT": "manual"})
        with self.assertRaises(ValueError):
            info.set_labels(["NOT_A_BONE"], "source")
        with self.assertRaises(ValueError):
            info.set_labels(["FEMUR_LEFT"], {})  # no origin for the label
        with self.assertRaises(ValidationError):
            FileInfo(source=1, source_scan_id="a", labels={"FEMUR_LEFT": "source"})  # an image has no labels

    def test_file_license(self):
        dataset = DatasetInfo(
            name="A",
            sources={
                1: SourceInfo(
                    name="A",
                    license=Licenses(image=RESTRICTED, segmentation="CC-BY-4.0"),
                    subsets={"b": SubsetInfo(name="B", license=Licenses(segmentation="CC-BY-SA-4.0"))},
                )
            },
        )
        self.assertEqual(file_license("image", FileInfo(source=1, source_scan_id="a"), dataset), RESTRICTED)
        self.assertEqual(file_license("segmentation", LabeledFileInfo(source=1, source_scan_id="a"), dataset), "CC-BY-4.0")
        in_subset = LabeledFileInfo(source=1, source_scan_id="a", subset="b")
        self.assertEqual(file_license("segmentation", in_subset, dataset), "CC-BY-SA-4.0")
        own = LabeledFileInfo(source=1, source_scan_id="a", subset="b", license="CC-BY-NC-4.0")
        self.assertEqual(file_license("segmentation", own, dataset), "CC-BY-NC-4.0")  # the file's own
        self.assertIsNone(file_license("segmentation", LabeledFileInfo(source=2, source_scan_id="a"), dataset))  # not a source
        self.assertIsNone(file_license("image", FileInfo(source=1, source_scan_id="a"), None))  # dataset.json not known
        with self.assertRaises(ValidationError):
            FileInfo(source=1, source_scan_id="a", license="CC BY")


class TestScanInfo(unittest.TestCase):
    def test_fields(self):
        scan = ScanInfo(weight=81, registered_to="scan-02", postmortem=True)
        self.assertEqual(scan.to_dict(), {"weight": 81.0, "postmortem": True, "registered_to": "scan-02"})
        for field, bad in (("registered_to", "2"), ("age", -1), ("weight", 0), ("postmortem", "yes"), ("image", "40")):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ScanInfo(**{field: bad})


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
            info = {"name": "A", "sources": {"1": {"name": "A"}}, "schema_version": "0.3.0"}
            (folder / layout.DATASET_JSON).write_text(json.dumps(info))
            with self.assertRaises(ValueError):
                Database(tmp).dataset(1).info


class TestLayout(unittest.TestCase):
    def test_names(self):
        self.assertEqual(layout.scan_key(1, 23, 4), "ds-0001/sub-000023/scan-04")
        self.assertEqual(layout.parse_scan_key("ds-0001/sub-000023/scan-04"), (1, 23, 4))
        self.assertEqual(layout.parse_subject_dir("sub-000023"), 23)
        for bad in ("sub-23", "sub-000000", "scan-01"):
            self.assertIsNone(layout.parse_subject_dir(bad))
        self.assertEqual(layout.dataset_dir_name(27), "ds-0027")
        self.assertEqual(layout.parse_dataset_dir("ds-0027"), 27)
        self.assertIsNone(layout.parse_dataset_dir("ds-027"))  # three digits
        self.assertEqual(layout.subject_dir_name(10_000), "sub-010000")
        for make, too_big in ((layout.dataset_dir_name, 10_000), (layout.subject_dir_name, 1_000_000)):
            with self.assertRaises(ValueError):
                make(too_big)

    def test_data_files(self):
        x = Path("x")
        self.assertEqual(layout.data_file(x, "image"), x / "image.nii.gz")
        self.assertEqual(layout.data_json(x, "image"), x / "image.json")
        self.assertEqual(layout.data_file(x, "segmentation", 2), x / "segmentation-2.nrrd")
        self.assertEqual(layout.data_json(x, "segmentation", 2), x / "segmentation-2.json")
        self.assertEqual(layout.model_zip(x, "nurbs", 1), x / "nurbs-1.zip")
        self.assertEqual(layout.model_folder(x, "mesh", 3), x / "mesh-3")
        for bad in (("image", 1), ("segmentation", None), ("segmentation", 0), ("landmarks", 1)):
            with self.subTest(file=bad), self.assertRaises(ValueError):
                layout.data_file(x, *bad)
        entries = {
            "image.nii.gz": ("image", None, "data"),
            "image.json": ("image", None, "json"),
            "segmentation-1.nrrd": ("segmentation", 1, "data"),
            "segmentation-12.json": ("segmentation", 12, "json"),
            "mesh-1.zip": ("mesh", 1, "data"),
            "mesh-1": ("mesh", 1, "folder"),
            "nurbs-2.json": ("nurbs", 2, "json"),
        }
        for name, parsed in entries.items():
            self.assertEqual(layout.parse_scan_entry(name), parsed)
        for other in ("scan.json", "mesh.zip", "segmentation-1.seg.nrrd", "segmentation-0.nrrd", "segmentation-01.nrrd",
                      "segmentation-1", "mesh-1.nrrd", "image-1.nii.gz"):
            self.assertIsNone(layout.parse_scan_entry(other), other)


if __name__ == "__main__":
    unittest.main()
