import fnmatch
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

import support
from bonehub_format import Licenses, layout, validate
from bonehub_format.index import build_index
from bonehub_format.publish import excluded_from_upload, write_hub_files
from bonehub_format.registry import parse_registry
from bonehub_format.webpage import build_webpage

REGISTRY_JSON = json.dumps(
    [{"BoneHub ID": i, "Dataset Name": name, "Country": [country]} for i, name, country in
     ((3, "Third", "NL"), (5, "Fifth", "US"), (6, "Sixth", "DE"), (7, "Seventh", "FR"))]
)


class TestDatabase(unittest.TestCase):
    """Four datasets: 1 from source 3; 2 from source 5 with segmentations of source 7; 3 restricted (source 6); 4 with
    restricted images and open segmentations (source 3)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.root = cls.tmp / "root"
        source_1 = support.make_source(cls.tmp / "source_1", {"p1": ["ct"], "p2": ["ct"]})
        support.SyntheticConverter(source_1).export_to_bonehub_format(cls.root, 1, num_workers=1, verbose=False)

        source_2 = support.make_source(cls.tmp / "source_2", {"q1": ["ct"]}, first_seed=10)
        support.SyntheticConverter(source_2, contributions=True, source_id=5).export_to_bonehub_format(
            cls.root, 2, num_workers=1, verbose=False
        )

        source_3 = support.make_source(cls.tmp / "source_3", {"r1": ["ct"]}, first_seed=20)
        support.SyntheticConverter(source_3, license="restricted", source_id=6).export_to_bonehub_format(
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
        self.assertEqual(registry, {3: "Third", 5: "Fifth", 6: "Sixth", 7: "Seventh"})
        self.assertEqual(validate(self.root, registry=registry), [])
        problems = validate(self.root, registry={3: "Third", 5: "Fifth", 6: "Sixth"})
        self.assertEqual(problems, [f"{self.root / 'ds-0002' / 'dataset.json'}: source 7 is not a BoneHub ID of the registry."])
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

    def test_2_malformed_folders_are_reported(self):
        old = self.root / "ds-001"
        old.mkdir()
        empty = self.root / "ds-0009"
        empty.mkdir()
        shutil.copy(self.root / "ds-0001" / "dataset.json", empty / "dataset.json")
        try:
            problems = validate(self.root)
            self.assertTrue(any("ds-001: not a dataset folder" in p for p in problems))
            self.assertTrue(any("ds-0009: has no subjects" in p for p in problems))
        finally:
            old.rmdir()
            shutil.rmtree(empty)

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
        self.assertEqual((first["sources"], first["image_source"], first["source_scan_id"], first["source_subject_id"]),
                         ([3], 3, "p1/ct", "p1"))
        self.assertEqual((first["image_license"], first["image_published"]), ("CC-BY-4.0", True))
        withheld = [row["dataset_id"] for row in scans if not row["image_published"]]
        self.assertEqual(withheld, [3, 4])
        merged = [row for row in scans if row["dataset_id"] == 2][0]
        self.assertEqual(merged["sources"], [5, 7])
        self.assertEqual(merged["segmentation"], ["GLUTEUS_MAXIMUS_RIGHT", "FEMUR_LEFT", "FEMUR_RIGHT"])

        files = pq.read_table(paths["files"]).to_pylist()
        first_files = [(row["kind"], row["number"], row["path"], row["source"], row["published"]) for row in files if row["scan"] == first["scan"]]
        self.assertEqual(first_files, [
            ("image", None, "ds-0001/sub-000001/scan-01/image.nii.gz", 3, True),
            ("segmentation", 1, "ds-0001/sub-000001/scan-01/segmentation-1.nrrd", 3, True),
            ("mesh", 1, "ds-0001/sub-000001/scan-01/mesh-1.zip", 3, True),
        ])
        self.assertTrue(all(row["size"] > 0 for row in files))
        merged_files = [(row["path"].rsplit("/", 1)[1], row["source"], row["license"]) for row in files if row["dataset_id"] == 2]
        self.assertIn(("segmentation-2.nrrd", 7, "CC-BY-NC-4.0"), merged_files)

        labels = pq.read_table(paths["labels"]).to_pylist()
        femur = [row for row in labels if row["scan"] == first["scan"] and row["label"] == "FEMUR_LEFT"]
        self.assertEqual({(row["kind"], row["number"], row["source"], row["origin"], row["truncated"]) for row in femur},
                         {("segmentation", 1, 3, "source", True), ("mesh", 1, 3, "derived:test-mesher@1.0", None)})
        published = {(row["dataset_id"], row["kind"]): row["published"] for row in labels}
        self.assertEqual((published[(3, "segmentation")], published[(4, "segmentation")]), (False, True))

        datasets = pq.read_table(paths["datasets"]).to_pylist()
        self.assertEqual(
            [(row["dataset_id"], row["sources"], row["subjects"], row["scans"]) for row in datasets],
            [(1, [3], 2, 2), (2, [5, 7], 1, 1), (3, [6], 1, 1), (4, [3], 1, 1)],
        )
        sources = pq.read_table(paths["sources"]).to_pylist()
        self.assertEqual([(row["dataset_id"], row["source_id"], row["files"], row["scans"]) for row in sources],
                         [(1, 3, 6, 2), (2, 5, 3, 1), (2, 7, 1, 1), (3, 6, 3, 1), (4, 3, 3, 1)])
        self.assertEqual((sources[4]["license_image"], sources[4]["license_segmentation"]), ("restricted", "CC-BY-4.0"))

    def test_4_hub_files_and_webpage(self):
        build_index(self.root)
        write_hub_files(self.root, "Org/Test-Dataset", "Test Dataset", homepage="https://example.org")
        self.assertEqual(validate(self.root, publishing=True, registry=parse_registry(REGISTRY_JSON)), [])
        card = (self.root / "README.md").read_text(encoding="utf-8")
        self.assertTrue(card.startswith("---\npretty_name: \"Test Dataset\"\nlicense: other\n"))
        for table in ("scans", "files", "labels", "datasets", "sources"):
            self.assertIn(f"data_files: index/{table}.parquet", card)
        self.assertIn("- ds-0003, source 6 (Synthetic): 1 image, 1 segmentation, 1 mesh files", card)
        self.assertIn("- ds-0004, source 3 (Synthetic): 1 image files", card)
        self.assertIn("| ds-0002 | Synthetic | [5](https://bonehub.github.io/Public-Datasets/#dataset=5) Synthetic<br>"
                      "[7](https://bonehub.github.io/Public-Datasets/#dataset=7) Synthetic provider |", card)
        self.assertIn("resolve/main/<path>?download=true", card)
        gitignore = (self.root / ".gitignore").read_text(encoding="utf-8")
        for line in ("upload.md", "logs/", "**/mesh-*/", "/ds-0003/*/*/image.nii.gz", "/ds-0004/*/*/image.nii.gz"):
            self.assertIn(f"\n{line}\n", gitignore)
        self.assertIn("python -m bonehub_format upload", (self.root / "upload.md").read_text(encoding="utf-8"))
        license_file = (self.root / "LICENSE.md").read_text(encoding="utf-8")
        self.assertIn("restricted", license_file.lower())
        self.assertIn("| ds-0002 | [7](https://bonehub.github.io/Public-Datasets/#dataset=7) | Synthetic provider |", license_file)

        patterns = [pattern for _, pattern in excluded_from_upload(self.root)]
        def uploaded(path):
            return not any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
        for path in ("upload.md", "logs/ds-0001.log", "ds-0001/sub-000001/scan-01/mesh-1/FEMUR_LEFT.stl",
                     "ds-0003/sub-000001/scan-01/image.nii.gz", "ds-0003/sub-000001/scan-01/segmentation-1.nrrd",
                     "ds-0003/sub-000001/scan-01/mesh-1.zip", "ds-0004/sub-000001/scan-01/image.nii.gz"):
            self.assertFalse(uploaded(path), path)
        for path in ("README.md", "ds-0001/sub-000001/scan-01/mesh-1.zip", "ds-0001/sub-000001/scan-01/image.nii.gz",
                     "ds-0001/sub-000001/scan-01/mesh-1.json", "index/scans.parquet", "github-webpage/index.html",
                     "ds-0003/dataset.json", "ds-0003/sub-000001/scan-01/image.json",
                     "ds-0004/sub-000001/scan-01/segmentation-1.nrrd", "ds-0004/sub-000001/scan-01/mesh-1.zip"):
            self.assertTrue(uploaded(path), path)

        folder = build_webpage(self.root, "Test Dataset", hf_repo_id="Org/Test-Dataset", contact_html="<p>someone</p>")
        page = (folder / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("{{", page)
        self.assertIn("<h1>Test Dataset</h1>", page)
        self.assertIn("https://huggingface.co/datasets/Org/Test-Dataset", page)
        self.assertIn("the main source of this page", page)
        self.assertNotIn("Getting started", page)
        data = (folder / "data.js").read_text(encoding="utf-8")
        site = json.loads(data.split("const site = ", 1)[1].rsplit(";", 1)[0])
        self.assertEqual(site["repo"], "Org/Test-Dataset")
        self.assertEqual([d["folder"] for d in site["datasets"]], ["ds-0001", "ds-0002", "ds-0003", "ds-0004"])
        self.assertEqual(site["datasets"][1]["sources"], [[5, "Synthetic", None], [7, "Synthetic provider", None]])
        self.assertEqual(len(site["scans"]), 5)
        key, dataset, sex, age, modality, labels, files = site["scans"][0]
        self.assertEqual((key, dataset, sex, age, modality), ("ds-0001/sub-000001/scan-01", 0, "F", 60, "CT"))
        self.assertEqual([site["labels"][i] for i in labels], ["FEMUR_LEFT", "FEMUR_RIGHT"])
        self.assertEqual(site["sections"][site["labelSection"][labels[0]]], "Leg")
        self.assertEqual([(f[0], f[1], f[3], f[4], f[5]) for f in files],
                         [("image", None, True, 3, "CC-BY-4.0"), ("segmentation", 1, True, 3, "CC-BY-4.0"), ("mesh", 1, True, 3, "CC-BY-4.0")])
        self.assertTrue(all(f[2] > 0 for f in files))
        restricted = [scan for scan in site["scans"] if scan[0].startswith("ds-0003")][0]
        self.assertEqual({f[3] for f in restricted[6]}, {False})
        self.assertEqual(site["sources"]["7"]["licenses"]["segmentation"], "CC-BY-NC-4.0")
        self.assertNotIn("{{", (folder / "update_data.py").read_text(encoding="utf-8"))
        self.assertTrue((folder / ".nojekyll").exists() and (folder / "app.js").exists())


if __name__ == "__main__":
    unittest.main()
