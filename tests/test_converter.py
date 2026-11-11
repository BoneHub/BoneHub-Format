import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import SimpleITK as sitk

import support
from bonehub_format import Root, layout, pack, unpack, validate
from bonehub_format.converter import DatasetConversionError, natural_key


class ConvertedDataset(unittest.TestCase):
    """A synthetic source converted once, in worker processes, for the tests of this class."""

    SUBJECTS = {"case_10": ["ct"], "case_2": ["ct", "mr"], "case_1": ["ct"]}

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.source = support.make_source(cls.tmp / "source", cls.SUBJECTS)
        cls.root = cls.tmp / "root"
        cls.log = support.SyntheticConverter(cls.source).export_to_bonehub_format(cls.root, 3, num_workers=2, verbose=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_subjects_are_numbered_in_natural_order_of_source_id(self):
        dataset = Root(self.root).dataset(3)
        self.assertEqual([s.info.source_subject_id for s in dataset.subjects()], ["case_1", "case_2", "case_10"])
        self.assertEqual([scan.info.source_scan_id for scan in dataset.subject(2).scans()], ["case_2/ct", "case_2/mr"])
        self.assertEqual(natural_key("case_2") < natural_key("case_10"), True)

    def test_layout_and_metadata(self):
        scan_folder = self.root / "ds-003" / "sub-000002" / "scan-02"
        self.assertEqual(
            sorted(p.name for p in scan_folder.iterdir()),
            ["image.nii.gz", "mesh", "mesh.zip", "scan.json", "segmentation.seg.nrrd"],
        )
        info = json.loads((scan_folder / "scan.json").read_text())
        self.assertEqual(info["segmentation"], {"FEMUR_LEFT": "source", "FEMUR_RIGHT": "source"})
        self.assertEqual(info["mesh"], {"FEMUR_LEFT": "derived:test-mesher@1.0"})
        self.assertEqual(info["registered_to"], "scan-01")
        self.assertNotIn("subject_id", info)  # the folder name is the ID
        dataset_info = json.loads((self.root / "ds-003" / "dataset.json").read_text())
        self.assertEqual(dataset_info["schema_version"], "1.0")
        self.assertEqual(dataset_info["annotated_labels"], ["FEMUR_LEFT", "FEMUR_RIGHT"])
        self.assertTrue(self.log.parent == self.root / layout.LOGS_DIR)

    def test_integer_ct_is_stored_as_int16(self):
        image = sitk.ReadImage(str(self.root / "ds-003" / "sub-000001" / "scan-01" / "image.nii.gz"))
        self.assertEqual(image.GetPixelID(), sitk.sitkInt16)
        source = sitk.ReadImage(str(self.source / "case_1" / "ct" / "image.nii.gz"))
        self.assertEqual(sitk.GetArrayFromImage(image).tolist(), sitk.GetArrayFromImage(source).tolist())

    def test_mesh_zip_matches_the_folder(self):
        scan = Root(self.root).dataset(3).subject(1).scan(1)
        with zipfile.ZipFile(scan.path / "mesh.zip") as archive:
            self.assertEqual(archive.namelist(), ["FEMUR_LEFT.stl"])
        self.assertEqual(list(scan.model_files("mesh")), ["FEMUR_LEFT"])

    def test_validates(self):
        self.assertEqual(validate(self.root), [])


class TestConverterRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "root"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_failed_subject_leaves_no_folder_and_resume_converts_it(self):
        source = support.make_source(self.tmp / "source", {"a": ["ct"], support.FAIL_ON: ["ct"], "c": ["ct"]})
        converter = support.SyntheticConverter(source)
        with self.assertRaises(DatasetConversionError) as caught:
            converter.export_to_bonehub_format(self.root, 1, num_workers=1, verbose=False)
        self.assertEqual(caught.exception.failed_subject_ids, (3,))  # a, c, fail_me
        dataset_folder = self.root / "ds-001"
        self.assertEqual(sorted(p.name for p in dataset_folder.iterdir()), ["dataset.json", "sub-000001", "sub-000002"])

        (source / "c").rename(source / "b2")  # subject 2 is now another source subject: no resuming
        with self.assertRaises(DatasetConversionError) as caught:
            converter.export_to_bonehub_format(self.root, 1, resume=True, num_workers=1, verbose=False)
        self.assertIn("was converted from 'c'", str(caught.exception))
        (source / "b2").rename(source / "c")
        (source / support.FAIL_ON).rename(source / "z_fixed")  # no longer fails, and still sorts last
        written = (dataset_folder / "sub-000001" / "subject.json").stat().st_mtime_ns
        converter.export_to_bonehub_format(self.root, 1, resume=True, num_workers=1, verbose=False)
        self.assertEqual(len(Root(self.root).dataset(1)), 3)
        self.assertEqual((dataset_folder / "sub-000001" / "subject.json").stat().st_mtime_ns, written)  # kept

    def test_existing_dataset_needs_overwrite_and_overwrite_spares_other_folders(self):
        source = support.make_source(self.tmp / "source", {"a": ["ct"]})
        converter = support.SyntheticConverter(source)
        converter.export_to_bonehub_format(self.root, 1, num_workers=1, verbose=False)
        with self.assertRaises(FileExistsError):
            converter.export_to_bonehub_format(self.root, 1, num_workers=1, verbose=False)
        converter.export_to_bonehub_format(self.root, 1, overwrite=True, num_workers=1, verbose=False)
        other = self.root / "ds-002"
        other.mkdir()
        (other / "notes.txt").write_text("not a dataset")
        with self.assertRaises(FileExistsError):
            converter.export_to_bonehub_format(self.root, 2, overwrite=True, num_workers=1, verbose=False)
        self.assertTrue((other / "notes.txt").exists())

    def test_pack_unpack_and_validation_of_edits(self):
        source = support.make_source(self.tmp / "source", {"a": ["ct"]})
        support.SyntheticConverter(source).export_to_bonehub_format(self.root, 1, num_workers=1, verbose=False)
        scan_folder = self.root / "ds-001" / "sub-000001" / "scan-01"
        zip_bytes = (scan_folder / "mesh.zip").read_bytes()
        self.assertEqual(pack(self.root), [])  # up to date
        (scan_folder / "mesh" / "FEMUR_LEFT.stl").write_bytes(b"solid changed\nendsolid changed\n")
        self.assertTrue(any("run pack" in p for p in validate(self.root)))
        self.assertEqual(pack(self.root), [scan_folder / "mesh.zip"])
        self.assertEqual(validate(self.root), [])

        shutil.rmtree(scan_folder / "mesh")  # as downloaded: only the zip
        self.assertEqual(list(Root(self.root).dataset(1).subject(1).scan(1).model_files("mesh")), ["FEMUR_LEFT"])
        self.assertEqual(validate(self.root), [])
        self.assertEqual(unpack(self.root), [scan_folder / "mesh"])
        (scan_folder / "mesh" / "FEMUR_LEFT.stl").write_bytes(b"solid FEMUR_LEFT\nendsolid FEMUR_LEFT\n")
        pack(self.root)
        self.assertEqual((scan_folder / "mesh.zip").read_bytes(), zip_bytes)  # the same files give the same zip

        (scan_folder / "mesh" / "NOT_A_BONE.stl").write_bytes(b"")
        self.assertTrue(any("NOT_A_BONE" in p for p in validate(self.root)))


if __name__ == "__main__":
    unittest.main()
