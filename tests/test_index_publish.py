import fnmatch
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

import support
from bonehub_format import DatasetInfo, Licenses, layout, validate
from bonehub_format.index import build_index
from bonehub_format.publish import excluded_from_upload, write_hub_files
from bonehub_format.registry import parse_registry
from bonehub_format.webpage import build_webpage

REGISTRY_JSON = json.dumps(
    [{"BoneHub ID": i, "Dataset Name": name, "Country": [country]} for i, name, country in
     ((1, "First", "NL"), (2, "Second", "US"), (3, "Third", "DE"), (4, "Fourth", "FR"))]
)


class TestDatabase(unittest.TestCase):
    """Four datasets: 2 is made from 1, 3 is restricted, 4 has restricted images and open segmentations."""

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

        source_4 = support.make_source(cls.tmp / "source_4", {"s1": ["ct"]}, first_seed=30)
        licenses = Licenses(image="restricted", segmentation="CC-BY-4.0", mesh="CC-BY-4.0")
        support.SyntheticConverter(source_4, license=licenses).export_to_bonehub_format(cls.root, 4, num_workers=1, verbose=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_1_registry(self):
        registry = parse_registry(REGISTRY_JSON)
        self.assertEqual(registry, {1: "First", 2: "Second", 3: "Third", 4: "Fourth"})
        self.assertEqual(validate(self.root, registry=registry), [])
        problems = validate(self.root, registry={2: "Second", 3: "Third", 4: "Fourth"})
        self.assertEqual(len(problems), 2)  # ds-0001 itself, and derived_from of ds-0002
        self.assertTrue(any("ds-0001: 1 is not a BoneHub ID" in p for p in problems))
        self.assertTrue(any("derived_from names 1" in p for p in problems))
        bad_registries = (
            "not json",
            '{"BoneHub ID": 1, "Dataset Name": "A"}',  # not an array
            '[{"BoneHub ID": 1, "Name": "A"}]',  # no Dataset Name
            '[{"BoneHub ID": 1, "Dataset Name": "A"}, {"BoneHub ID": 1, "Dataset Name": "B"}]',
            '[{"BoneHub ID": "1", "Dataset Name": "A"}]',
            '[{"BoneHub ID": 0, "Dataset Name": "A"}]',
        )
        for bad in bad_registries:
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
        self.assertEqual(len(scans), 5)
        first = scans[0]
        self.assertEqual(first["scan"], "ds-0001/sub-000001/scan-01")
        self.assertEqual(first["image_size"], [16, 16, 12])
        self.assertEqual(first["image_spacing"], [0.8, 0.8, 2.0])
        self.assertEqual(first["segmentation"], ["FEMUR_LEFT", "FEMUR_RIGHT"])
        self.assertEqual(first["truncated"], ["FEMUR_LEFT"])
        self.assertEqual(first["gender"], "F")
        self.assertNotIn("subject_group", first)
        self.assertEqual((first["providers"], first["image_provider"], first["source_scan_id"]), ([1], 1, "p1/ct"))
        self.assertEqual((first["image_license"], first["image_published"]), ("CC-BY-4.0", True))
        withheld = [row["dataset_id"] for row in scans if not row["image_published"]]
        self.assertEqual(withheld, [3, 4])
        labels = pq.read_table(paths["labels"]).to_pylist()
        femur = [row for row in labels if row["scan"] == first["scan"] and row["label"] == "FEMUR_LEFT"]
        self.assertEqual({(row["kind"], row["provider"], row["origin"], row["truncated"]) for row in femur},
                         {("segmentation", 1, "source", True), ("mesh", 1, "derived:test-mesher@1.0", None)})
        published = {(row["dataset_id"], row["kind"]): row["published"] for row in labels}
        self.assertEqual((published[(3, "segmentation")], published[(4, "segmentation")]), (False, True))
        datasets = pq.read_table(paths["datasets"]).to_pylist()
        self.assertEqual(
            [(row["dataset_id"], row["subjects"], row["scans"], row["scans_provided"]) for row in datasets],
            [(1, 2, 2, 2), (2, 1, 1, 1), (3, 1, 1, 1), (4, 1, 1, 1)],
        )
        self.assertEqual([row["derived_from"] for row in datasets], [None, [1], None, None])
        self.assertEqual((datasets[3]["license_image"], datasets[3]["license_segmentation"]), ("restricted", "CC-BY-4.0"))

    def test_4_hub_files_and_webpage(self):
        build_index(self.root)
        write_hub_files(self.root, "Org/Test-Dataset", "Test Dataset", homepage="https://example.org")
        self.assertEqual(validate(self.root, publishing=True, registry=parse_registry(REGISTRY_JSON)), [])
        card = (self.root / "README.md").read_text(encoding="utf-8")
        self.assertTrue(card.startswith("---\npretty_name: \"Test Dataset\"\nlicense: other\n"))
        self.assertIn("data_files: index/scans.parquet", card)
        self.assertIn("- 3 (Synthetic): 1 image, 1 segmentation, 1 mesh files", card)
        self.assertIn("- 4 (Synthetic): 1 image files", card)
        self.assertIn("| [1](https://bonehub.github.io/Public-Datasets/#dataset=1) | Synthetic |", card)
        gitignore = (self.root / ".gitignore").read_text(encoding="utf-8")
        for line in ("upload.md", "logs/", "**/mesh.ds-*/", "/ds-0003/*/*/image.nii.gz", "/ds-0004/*/*/image.nii.gz"):
            self.assertIn(f"\n{line}\n", gitignore)
        self.assertIn("python -m bonehub_format upload", (self.root / "upload.md").read_text(encoding="utf-8"))
        self.assertIn("restricted", (self.root / "LICENSE.md").read_text(encoding="utf-8").lower())

        patterns = [pattern for _, pattern in excluded_from_upload(self.root)]
        def uploaded(path):
            return not any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
        for path in ("upload.md", "logs/ds-0001.log", "ds-0001/sub-000001/scan-01/mesh.ds-0001/FEMUR_LEFT.stl",
                     "ds-0003/sub-000001/scan-01/image.nii.gz", "ds-0003/sub-000001/scan-01/segmentation.ds-0003.seg.nrrd",
                     "ds-0003/sub-000001/scan-01/mesh.ds-0003.zip", "ds-0004/sub-000001/scan-01/image.nii.gz"):
            self.assertFalse(uploaded(path), path)
        for path in ("README.md", "ds-0001/sub-000001/scan-01/mesh.ds-0001.zip", "ds-0001/sub-000001/scan-01/image.nii.gz",
                     "index/scans.parquet", "github-webpage/index.html", "ds-0003/dataset.json",
                     "ds-0004/sub-000001/scan-01/segmentation.ds-0004.seg.nrrd", "ds-0004/sub-000001/scan-01/mesh.ds-0004.zip"):
            self.assertTrue(uploaded(path), path)

        folder = build_webpage(self.root, "Test Dataset", hf_repo_id="Org/Test-Dataset", contact_html="<p>someone</p>")
        page = (folder / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("{{", page)
        self.assertIn("<h1>Test Dataset</h1>", page)
        self.assertIn("https://huggingface.co/datasets/Org/Test-Dataset", page)
        data = (folder / "data.js").read_text(encoding="utf-8")

        def variable(name):
            return json.loads(data.split(f"const {name} = ", 1)[1].split(";\n", 1)[0])

        scans = variable("scansData")
        self.assertEqual(len(scans), 5)
        self.assertEqual(scans[0]["Segmentation Available"], "FEMUR_LEFT; FEMUR_RIGHT")
        structures = variable("structuresData")
        femur_left = [i for i, group in enumerate(structures["groups"]) if group["id"] == "femur-L"][0]
        self.assertIn(femur_left, scans[0]["_groups"])  # the scan's groups, for the coverage figure's filter
        self.assertEqual(variable("coverageData")["femur-L"]["scans"], 5)
        cards = variable("datasetCards")
        self.assertEqual(cards[3]["kinds"]["image"], {"license": "restricted", "published": 0, "withheld": 1})
        datasets = variable("datasetsData")
        self.assertEqual(datasets[1]["Registry"], "https://bonehub.github.io/Public-Datasets/#dataset=2")
        self.assertNotIn("{{", (folder / "update_data.py").read_text(encoding="utf-8"))
        self.assertTrue((folder / ".nojekyll").exists() and (folder / "update_data.py").exists())


if __name__ == "__main__":
    unittest.main()
