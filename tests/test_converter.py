import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import SimpleITK as sitk

import support
from bonehub_format import Database, layout, pack, unpack, validate
from bonehub_format.converter import DatasetConversionError, natural_key
from bonehub_format.segmentation_file import read_segmentation


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
        dataset = Database(self.root).dataset(3)
        self.assertEqual([s.info.source_subject_id(3) for s in dataset.subjects()], ["case_1", "case_2", "case_10"])
        self.assertEqual([scan.info.source_scan_id() for scan in dataset.subject(2).scans()], ["case_2/ct", "case_2/mr"])
        self.assertEqual(natural_key("case_2") < natural_key("case_10"), True)

    def test_layout_and_metadata(self):
        scan_folder = self.root / "ds-0003" / "sub-000002" / "scan-02"
        self.assertEqual(
            sorted(p.name for p in scan_folder.iterdir()),
            ["image.nii.gz", "mesh.ds-0003", "mesh.ds-0003.zip", "scan.json", "segmentation.ds-0003.seg.nrrd"],
        )
        info = json.loads((scan_folder / "scan.json").read_text())
        self.assertEqual(info["image"], "ds-0003")
        provider = info["providers"]["ds-0003"]
        self.assertEqual(provider["source_scan_id"], "case_2/mr")
        self.assertEqual(provider["segmentation"], {"FEMUR_LEFT": "source", "FEMUR_RIGHT": "source"})
        self.assertEqual(provider["mesh"], {"FEMUR_LEFT": "derived:test-mesher@1.0"})
        self.assertEqual(info["registered_to"], "scan-01")
        subject = json.loads((self.root / "ds-0003" / "sub-000002" / "subject.json").read_text())
        self.assertEqual(subject, {"sources": {"ds-0003": "case_2"}, "gender": "F"})
        self.assertNotIn("subject_id", info)  # the folder name is the ID
        dataset_info = json.loads((self.root / "ds-0003" / "dataset.json").read_text())
        self.assertEqual(dataset_info["schema_version"], "1.0")
        self.assertEqual(dataset_info["annotated_labels"], ["FEMUR_LEFT", "FEMUR_RIGHT"])
        self.assertTrue(self.log.parent == self.root / layout.LOGS_DIR)

    def test_integer_ct_is_stored_as_int16(self):
        image = sitk.ReadImage(str(self.root / "ds-0003" / "sub-000001" / "scan-01" / "image.nii.gz"))
        self.assertEqual(image.GetPixelID(), sitk.sitkInt16)
        source = sitk.ReadImage(str(self.source / "case_1" / "ct" / "image.nii.gz"))
        self.assertEqual(sitk.GetArrayFromImage(image).tolist(), sitk.GetArrayFromImage(source).tolist())

    def test_mesh_zip_matches_the_folder(self):
        scan = Database(self.root).dataset(3).subject(1).scan(1)
        with zipfile.ZipFile(scan.path / "mesh.ds-0003.zip") as archive:
            self.assertEqual(archive.namelist(), ["FEMUR_LEFT.stl"])
        self.assertEqual(list(scan.model_files("mesh", 3)), ["FEMUR_LEFT"])

    def test_validates(self):
        self.assertEqual(validate(self.root), [])


class TestMergedScans(unittest.TestCase):
    """A dataset whose scans hold the segmentation another dataset (the provider) made of the same images."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        source = support.make_source(cls.tmp / "source", {"a": ["ct"], "b": ["ct"]})
        cls.root = cls.tmp / "root"
        support.SyntheticConverter(source, contributions=True).export_to_bonehub_format(cls.root, 3, num_workers=1, verbose=False)
        support.ProviderConverter(cls.tmp / "provider").export_to_bonehub_format(cls.root, support.PROVIDER_ID, verbose=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_each_provider_has_its_own_files_and_entry(self):
        scan = Database(self.root).dataset(3).subject(1).scan(1)
        self.assertEqual(scan.provider_ids, [3, support.PROVIDER_ID])
        self.assertEqual(sorted(scan.segmentation_paths()), [3, support.PROVIDER_ID])
        self.assertTrue((scan.path / "segmentation.ds-0007.seg.nrrd").is_file())
        provider = scan.info.provider(support.PROVIDER_ID)
        self.assertEqual(provider.source_scan_id, "provider-a/ct")
        self.assertEqual(provider.segmentation, {"FEMUR_LEFT": "model:TestModel@2.0", "GLUTEUS_MAXIMUS_RIGHT": "model:TestModel@2.0"})
        self.assertEqual(scan.info.labels("segmentation"), ["GLUTEUS_MAXIMUS_RIGHT", "FEMUR_LEFT", "FEMUR_RIGHT"])  # label map order
        self.assertEqual(scan.info.labels("segmentation", 3), ["FEMUR_LEFT", "FEMUR_RIGHT"])
        self.assertEqual(scan.info.image, "ds-0003")
        self.assertEqual(scan.subject.info.sources, {"ds-0003": "a", "ds-0007": "provider-a"})

    def test_licenses_follow_the_provider(self):
        scan = Database(self.root).dataset(3).subject(1).scan(1)
        self.assertEqual(scan.license("image"), "CC-BY-4.0")
        self.assertEqual(scan.license("segmentation", 3), "CC-BY-4.0")
        self.assertEqual(scan.license("segmentation", support.PROVIDER_ID), "CC-BY-NC-4.0")

    def test_provider_without_subjects_is_its_dataset_json(self):
        self.assertEqual(sorted(p.name for p in (self.root / "ds-0007").iterdir()), ["dataset.json"])
        self.assertEqual(validate(self.root), [])

    def _convert_with_moved_labels(self, name: str, move) -> Path:
        """Convert a source whose provider labels are the home labels written by move(labels) -> sitk.Image."""
        source = support.make_source(self.tmp / f"{name}-source", {"a": ["ct"]})
        labels = sitk.ReadImage(str(source / "a" / "ct" / "labels.nii.gz"))
        sitk.WriteImage(move(labels), str(source / "a" / "ct" / "labels_moved.nii.gz"))
        converter = support.SyntheticConverter(source, contributions=True)
        converter.custom_data_handlers.read_dataset = support.read_dataset_with_moved_contribution
        root = self.tmp / f"{name}-root"
        converter.export_to_bonehub_format(root, 3, num_workers=1, verbose=False)
        return root

    def test_a_provider_mask_on_another_grid_fails_the_subject(self):
        def shift(labels):
            labels.SetOrigin(tuple(o + 5.0 for o in labels.GetOrigin()))
            return labels

        with self.assertRaises(DatasetConversionError):
            self._convert_with_moved_labels("shifted", shift)
        self.assertFalse((self.tmp / "shifted-root" / "ds-0003" / "sub-000001").exists())

    def _assert_on_the_image_grid(self, root: Path):
        scan = Database(root).dataset(3).subject(1).scan(1)
        image = sitk.ReadImage(str(scan.path / layout.IMAGE_FILE))
        home, provider = (read_segmentation(scan.segmentation_paths()[i]) for i in (3, support.PROVIDER_ID))
        for mask in (home, provider):
            self.assertEqual(mask.GetSize(), image.GetSize())
            self.assertEqual(mask.GetOrigin(), image.GetOrigin())
            self.assertEqual(mask.GetDirection(), image.GetDirection())
        # the same voxels are labelled: the provider's labels are the home labels under other names
        self.assertTrue((sitk.GetArrayFromImage(home) > 0).sum() > 0)
        self.assertTrue(((sitk.GetArrayFromImage(home) > 0) == (sitk.GetArrayFromImage(provider) > 0)).all())

    def test_a_provider_mask_stored_flipped_is_put_on_the_image_grid(self):
        def flip(labels):
            flipped = sitk.Flip(labels, [True, False, False])  # the same voxels, the first axis stored reversed
            self.assertNotEqual(flipped.GetDirection(), labels.GetDirection())
            return flipped

        self._assert_on_the_image_grid(self._convert_with_moved_labels("flipped", flip))

    def test_a_provider_mask_with_an_origin_off_by_less_than_half_a_voxel_is_put_on_the_image_grid(self):
        def nudge(labels):
            labels.SetOrigin(tuple(o + 0.4 * s for o, s in zip(labels.GetOrigin(), labels.GetSpacing())))
            return labels

        self._assert_on_the_image_grid(self._convert_with_moved_labels("nudged", nudge))

    def test_a_provider_missing_from_the_database_is_a_problem(self):
        root = self.tmp / "without-provider"
        shutil.copytree(self.root, root)
        shutil.rmtree(root / "ds-0007")
        self.assertTrue(any("provider ds-0007 has no dataset folder" in p for p in validate(root)))


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
        dataset_folder = self.root / "ds-0001"
        self.assertEqual(sorted(p.name for p in dataset_folder.iterdir()), ["dataset.json", "sub-000001", "sub-000002"])

        (source / "c").rename(source / "b2")  # subject 2 is now another source subject: no resuming
        with self.assertRaises(DatasetConversionError) as caught:
            converter.export_to_bonehub_format(self.root, 1, resume=True, num_workers=1, verbose=False)
        self.assertIn("was converted from 'c'", str(caught.exception))
        (source / "b2").rename(source / "c")
        (source / support.FAIL_ON).rename(source / "z_fixed")  # no longer fails, and still sorts last
        written = (dataset_folder / "sub-000001" / "subject.json").stat().st_mtime_ns
        converter.export_to_bonehub_format(self.root, 1, resume=True, num_workers=1, verbose=False)
        self.assertEqual(len(Database(self.root).dataset(1)), 3)
        self.assertEqual((dataset_folder / "sub-000001" / "subject.json").stat().st_mtime_ns, written)  # kept

    def test_existing_dataset_needs_overwrite_and_overwrite_spares_other_folders(self):
        source = support.make_source(self.tmp / "source", {"a": ["ct"]})
        converter = support.SyntheticConverter(source)
        converter.export_to_bonehub_format(self.root, 1, num_workers=1, verbose=False)
        with self.assertRaises(FileExistsError):
            converter.export_to_bonehub_format(self.root, 1, num_workers=1, verbose=False)
        converter.export_to_bonehub_format(self.root, 1, overwrite=True, num_workers=1, verbose=False)
        other = self.root / "ds-0002"
        other.mkdir()
        (other / "notes.txt").write_text("not a dataset")
        with self.assertRaises(FileExistsError):
            converter.export_to_bonehub_format(self.root, 2, overwrite=True, num_workers=1, verbose=False)
        self.assertTrue((other / "notes.txt").exists())

    def test_pack_unpack_and_validation_of_edits(self):
        source = support.make_source(self.tmp / "source", {"a": ["ct"]})
        support.SyntheticConverter(source).export_to_bonehub_format(self.root, 1, num_workers=1, verbose=False)
        scan_folder = self.root / "ds-0001" / "sub-000001" / "scan-01"
        mesh_folder, mesh_zip = scan_folder / "mesh.ds-0001", scan_folder / "mesh.ds-0001.zip"
        zip_bytes = mesh_zip.read_bytes()
        self.assertEqual(pack(self.root), [])  # up to date
        (mesh_folder / "FEMUR_LEFT.stl").write_bytes(b"solid changed\nendsolid changed\n")
        self.assertTrue(any("run pack" in p for p in validate(self.root)))
        self.assertEqual(pack(self.root), [mesh_zip])
        self.assertEqual(validate(self.root), [])

        shutil.rmtree(mesh_folder)  # as downloaded: only the zip
        self.assertEqual(list(Database(self.root).dataset(1).subject(1).scan(1).model_files("mesh", 1)), ["FEMUR_LEFT"])
        self.assertEqual(validate(self.root), [])
        self.assertEqual(unpack(self.root), [mesh_folder])
        (mesh_folder / "FEMUR_LEFT.stl").write_bytes(b"solid FEMUR_LEFT\nendsolid FEMUR_LEFT\n")
        pack(self.root)
        self.assertEqual(mesh_zip.read_bytes(), zip_bytes)  # the same files give the same zip

        (mesh_folder / "NOT_A_BONE.stl").write_bytes(b"")
        self.assertTrue(any("NOT_A_BONE" in p for p in validate(self.root)))


if __name__ == "__main__":
    unittest.main()
