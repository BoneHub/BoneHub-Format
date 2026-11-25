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
    Database,
    Licenses,
    ProviderInfo,
    ScanInfo,
    SubjectInfo,
    SubsetInfo,
    file_license,
    is_compatible_schema_version,
    layout,
)


class TestDatasetInfo(unittest.TestCase):
    def test_license_per_kind(self):
        info = DatasetInfo(name="A", license=Licenses(image=RESTRICTED, segmentation="CC-BY-4.0"))
        self.assertEqual((info.license_of("image"), info.license_of("segmentation"), info.license_of("mesh")), ("restricted", "CC-BY-4.0", None))
        self.assertEqual(Licenses.all("MIT").to_dict(), {"image": "MIT", "segmentation": "MIT", "mesh": "MIT", "nurbs": "MIT"})
        for bad in ("CC BY 4.0", "cc-by-4.0", "Other", ""):
            with self.subTest(license=bad), self.assertRaises(ValidationError):
                Licenses(image=bad)
        with self.assertRaises(ValidationError):
            DatasetInfo(name="A", license="CC-BY-4.0")  # a license is given per kind
        self.assertIn("CC-BY-NC-3.0", LICENSES)

    def test_subset_license_overrides_the_dataset(self):
        info = DatasetInfo(
            name="CADS",
            license=Licenses.all("CC-BY-4.0"),
            subsets={"0004_lits": SubsetInfo(name="LiTS", license=Licenses(image="CC-BY-NC-SA-4.0"))},
        )
        self.assertEqual(info.license_of("image", "0004_lits"), "CC-BY-NC-SA-4.0")
        self.assertEqual(info.license_of("segmentation", "0004_lits"), "CC-BY-4.0")  # the subset gives none
        self.assertEqual(info.license_of("image", "unknown"), "CC-BY-4.0")

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
            self.assertEqual(SubjectInfo(sources={"ds-0001": "1"}, gender=value).gender, value)
        for bad in ("", "male", "m"):
            with self.subTest(gender=bad), self.assertRaises(ValidationError):
                SubjectInfo(sources={"ds-0001": "1"}, gender=bad)

    def test_sources_are_keyed_and_sorted_by_dataset(self):
        info = SubjectInfo(sources={"ds-0134": "0001", "ds-0040": "PETCT_1"})
        self.assertEqual(list(info.sources), ["ds-0040", "ds-0134"])
        self.assertEqual(info.source_subject_id(134), "0001")
        self.assertIsNone(info.source_subject_id(18))
        for bad in ("40", "ds-40", "dataset-0040"):
            with self.subTest(key=bad), self.assertRaises(ValidationError):
                SubjectInfo(sources={bad: "1"})


class TestScanInfo(unittest.TestCase):
    def test_origins(self):
        good = ["source", "manual", "model:TotalSegmentator@2.4.0", "derived:mesh2nurbs-rhino3d@1.0"]
        for origin in good:
            ProviderInfo(source_scan_id="a", mesh={"FEMUR_LEFT": origin})
        for bad in ("Source", "model:TotalSegmentator", "derived:@1.0", "model:a b@1", 1):
            with self.subTest(origin=bad), self.assertRaises(ValidationError):
                ProviderInfo(source_scan_id="a", segmentation={"FEMUR_LEFT": bad})

    def test_labels_are_checked_and_kept_in_label_map_order(self):
        provider = ProviderInfo(source_scan_id="a")
        provider.set_labels("segmentation", ["TIBIA_LEFT", "FEMUR_LEFT"], "source")
        self.assertEqual(provider.labels("segmentation"), ["FEMUR_LEFT", "TIBIA_LEFT"])
        provider.set_labels("nurbs", ["FEMUR_LEFT"], {"FEMUR_LEFT": "manual"})
        self.assertEqual(provider.nurbs, {"FEMUR_LEFT": "manual"})
        provider.set_labels("mesh", [], "source")
        self.assertIsNone(provider.mesh)
        with self.assertRaises(ValueError):
            provider.set_labels("segmentation", ["NOT_A_BONE"], "source")
        with self.assertRaises(ValueError):
            provider.set_labels("mesh", ["FEMUR_LEFT"], {})  # no origin for the label
        with self.assertRaises(ValueError):
            provider.labels("landmarks")

    def test_labels_of_all_providers(self):
        scan = ScanInfo(
            image="ds-0040",
            providers={
                "ds-0134": ProviderInfo(source_scan_id="0001", segmentation={"FEMUR_LEFT": "source"}),
                "ds-0040": ProviderInfo(source_scan_id="PETCT_1"),
                "ds-0018": ProviderInfo(source_scan_id="AutoPET_1", segmentation={"SKULL": "source", "FEMUR_LEFT": "source"}),
            },
        )
        self.assertEqual(list(scan.providers), ["ds-0018", "ds-0040", "ds-0134"])
        self.assertEqual(scan.provider_ids, [18, 40, 134])
        self.assertEqual(scan.image_provider, 40)
        self.assertEqual(scan.labels("segmentation"), ["SKULL", "FEMUR_LEFT"])
        self.assertEqual(scan.labels("segmentation", 134), ["FEMUR_LEFT"])
        self.assertEqual(scan.labels("segmentation", 27), [])
        self.assertEqual(scan.source_scan_id(), "PETCT_1")
        with self.assertRaises(ValidationError):  # the image must come from a provider
            ScanInfo(image="ds-0027", providers={"ds-0040": ProviderInfo(source_scan_id="a")})

    def test_file_license(self):
        dataset = DatasetInfo(name="A", license=Licenses(image=RESTRICTED, segmentation="CC-BY-4.0"))
        scan = ScanInfo(
            image="ds-0001",
            providers={"ds-0001": ProviderInfo(source_scan_id="a", license=Licenses(segmentation="CC-BY-NC-4.0"))},
        )
        self.assertEqual(file_license(scan, "image", 1, dataset), RESTRICTED)
        self.assertEqual(file_license(scan, "segmentation", 1, dataset), "CC-BY-NC-4.0")  # the scan's own
        self.assertIsNone(file_license(scan, "segmentation", 2, dataset))  # not a provider
        self.assertIsNone(file_license(scan, "image", 1, None))  # provider's dataset.json not known

    def test_other_fields(self):
        scan = ScanInfo(weight=81, registered_to="scan-02", postmortem=True)
        self.assertEqual(scan.to_dict(), {"weight": 81.0, "postmortem": True, "registered_to": "scan-02", "providers": {}})
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
            (folder / layout.DATASET_JSON).write_text(json.dumps({"name": "A", "schema_version": "0.3.0"}))
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
        self.assertEqual(layout.model_zip(Path("x"), "nurbs", 40), Path("x") / "nurbs.ds-0040.zip")
        self.assertEqual(layout.model_folder(Path("x"), "mesh", 40), Path("x") / "mesh.ds-0040")
        self.assertEqual(layout.segmentation_file(Path("x"), 18), Path("x") / "segmentation.ds-0018.seg.nrrd")
        self.assertEqual(layout.parse_scan_entry("mesh.ds-0040.zip"), ("mesh", 40, True))
        self.assertEqual(layout.parse_scan_entry("nurbs.ds-0153"), ("nurbs", 153, False))
        self.assertEqual(layout.parse_scan_entry("segmentation.ds-0018.seg.nrrd"), ("segmentation", 18, False))
        for other in ("image.nii.gz", "mesh.zip", "segmentation.seg.nrrd", "mesh.ds-40"):
            self.assertIsNone(layout.parse_scan_entry(other))


if __name__ == "__main__":
    unittest.main()
