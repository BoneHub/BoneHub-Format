import fnmatch
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

import support
from bonehub_format import DatasetInfo, layout, validate
from bonehub_format.index import build_index
from bonehub_format.publish import excluded_from_upload, write_hub_files
from bonehub_format.registry import parse_registry
from bonehub_format.webpage import build_webpage

REGISTRY_CSV = "BoneHub ID,Dataset Name,Country\n1,First,NL\n2,Second,US\n3,Third,DE\n"


class TestDatabase(unittest.TestCase):
    """Three datasets: 2 is made from 1, 3 is restricted."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.root = cls.tmp / "root"
        source_1 = support.make_source(cls.tmp / "source_1", {"p1": ["ct"], "p2": ["ct"]})
        support.SyntheticConverter(source_1).export_to_bonehub_format(cls.root, 1, num_workers=1, verbose=False)

        source_2 = support.make_source(cls.tmp / "source_2", {"q1": ["ct"]}, first_seed=10)
        support.SyntheticConverter(source_2).export_to_bonehub_format(cls.root, 2, num_workers=1, verbose=False)
        dataset_json = cls.root / layout.dataset_dir_name(2) / layout.DATASET_JSON
        info = DatasetInfo.load(dataset_json)
        info.derived_from = [1]
        info.save(dataset_json)

        source_3 = support.make_source(cls.tmp / "source_3", {"r1": ["ct"]}, first_seed=20)
        support.SyntheticConverter(source_3, license="restricted").export_to_bonehub_format(
            cls.root, 3, num_workers=1, verbose=False
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_1_registry(self):
        registry = parse_registry(REGISTRY_CSV)
        self.assertEqual(registry, {1: "First", 2: "Second", 3: "Third"})
        self.assertEqual(validate(self.root, registry=registry), [])
        problems = validate(self.root, registry={2: "Second", 3: "Third"})
        self.assertEqual(len(problems), 2)  # ds-0001 itself, and derived_from of ds-0002
        self.assertTrue(any("ds-0001: 1 is not a BoneHub ID" in p for p in problems))
        self.assertTrue(any("derived_from names 1" in p for p in problems))
        for bad in ("BoneHub ID,Name\n1,A\n", "BoneHub ID,Dataset Name\n1,A\n1,B\n", "BoneHub ID,Dataset Name\nx,A\n"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_registry(bad)

    def test_2_malformed_folder_names_are_reported(self):
        old = self.root / "ds-001"
        old.mkdir()
        try:
            self.assertTrue(any("ds-001: not a dataset folder" in p for p in validate(self.root)))
        finally:
            old.rmdir()

    def test_3_index(self):
        paths = build_index(self.root)
        scans = pq.read_table(paths["scans"]).to_pylist()
        self.assertEqual(len(scans), 4)
        first = scans[0]
        self.assertEqual(first["scan"], "ds-0001/sub-000001/scan-01")
        self.assertEqual(first["image_size"], [16, 16, 12])
        self.assertEqual(first["image_spacing"], [0.8, 0.8, 2.0])
        self.assertEqual(first["segmentation"], ["FEMUR_LEFT", "FEMUR_RIGHT"])
        self.assertEqual(first["truncated"], ["FEMUR_LEFT"])
        self.assertEqual(first["gender"], "F")
        self.assertNotIn("subject_group", first)
        restricted = [row for row in scans if row["restricted"]]
        self.assertEqual([row["dataset_id"] for row in restricted], [3])
        labels = pq.read_table(paths["labels"]).to_pylist()
        femur = [row for row in labels if row["scan"] == first["scan"] and row["label"] == "FEMUR_LEFT"]
        self.assertEqual({(row["kind"], row["origin"], row["truncated"]) for row in femur},
                         {("segmentation", "source", True), ("mesh", "derived:test-mesher@1.0", None)})
        datasets = pq.read_table(paths["datasets"]).to_pylist()
        self.assertEqual([(row["dataset_id"], row["subjects"], row["scans"]) for row in datasets], [(1, 2, 2), (2, 1, 1), (3, 1, 1)])
        self.assertEqual([row["derived_from"] for row in datasets], [None, [1], None])

    def test_4_hub_files_and_webpage(self):
        build_index(self.root)
        write_hub_files(self.root, "Org/Test-Dataset", "Test Dataset", homepage="https://example.org")
        self.assertEqual(validate(self.root, publishing=True, registry=parse_registry(REGISTRY_CSV)), [])
        card = (self.root / "README.md").read_text(encoding="utf-8")
        self.assertTrue(card.startswith("---\npretty_name: \"Test Dataset\"\nlicense: other\n"))
        self.assertIn("data_files: index/scans.parquet", card)
        self.assertIn("Dataset 3 (Synthetic) is under a restricted license", card)
        gitignore = (self.root / ".gitignore").read_text(encoding="utf-8")
        for line in ("upload.md", "logs/", "**/mesh/", "/ds-0003/"):
            self.assertIn(f"\n{line}\n", gitignore)
        self.assertIn("python -m bonehub_format upload", (self.root / "upload.md").read_text(encoding="utf-8"))
        self.assertIn("restricted", (self.root / "LICENSE.md").read_text(encoding="utf-8").lower())

        patterns = [pattern for _, pattern in excluded_from_upload(self.root)]
        def uploaded(path):
            return not any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
        for path in ("upload.md", "logs/ds-0001.log", "ds-0001/sub-000001/scan-01/mesh/FEMUR_LEFT.stl", "ds-0003/dataset.json",
                     "ds-0003/sub-000001/scan-01/image.nii.gz"):
            self.assertFalse(uploaded(path), path)
        for path in ("README.md", "ds-0001/sub-000001/scan-01/mesh.zip", "ds-0001/sub-000001/scan-01/image.nii.gz",
                     "index/scans.parquet", "github-webpage/index.html", "ds-0031/dataset.json"):
            self.assertTrue(uploaded(path), path)

        folder = build_webpage(self.root, "Test Dataset", hf_repo_id="Org/Test-Dataset", contact_html="<p>someone</p>")
        page = (folder / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("{{", page)
        self.assertIn("<h1>Test Dataset</h1>", page)
        self.assertIn("https://huggingface.co/datasets/Org/Test-Dataset", page)
        data = (folder / "data.js").read_text(encoding="utf-8")
        scans = json.loads(data.split("const scansData = ", 1)[1].rstrip().rstrip(";"))
        self.assertEqual(len(scans), 4)
        self.assertEqual(scans[0]["Segmentation Available"], "FEMUR_LEFT; FEMUR_RIGHT")
        self.assertTrue((folder / ".nojekyll").exists() and (folder / "update_data.py").exists())


if __name__ == "__main__":
    unittest.main()
