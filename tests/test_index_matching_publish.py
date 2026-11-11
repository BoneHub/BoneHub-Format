import fnmatch
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import support
from bonehub_format import Link, Root, load_links, save_links, scan_groups, validate
from bonehub_format.index import build_index
from bonehub_format.matching import find_identical_images
from bonehub_format.publish import excluded_from_upload, write_hub_files
from bonehub_format.webpage import build_webpage


class TestRoot(unittest.TestCase):
    """Three datasets: 1 and 2 share an image (2 holds it flipped, cropped and as int16), 3 is restricted."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.root = cls.tmp / "root"
        source_1 = support.make_source(cls.tmp / "source_1", {"p1": ["ct"], "p2": ["ct"]})
        support.SyntheticConverter(source_1).export_to_bonehub_format(cls.root, 1, num_workers=1, verbose=False)

        source_2 = support.make_source(cls.tmp / "source_2", {"q1": ["ct"]}, first_seed=10)
        volume = support.ct_volume(0)  # the image of p1 in dataset 1
        support.write_image(volume[2:, :, ::-1], source_2 / "q1" / "ct" / "image.nii.gz", dtype=np.int16)
        support.SyntheticConverter(source_2).export_to_bonehub_format(cls.root, 2, num_workers=1, verbose=False)

        source_3 = support.make_source(cls.tmp / "source_3", {"r1": ["ct"]}, first_seed=20)
        support.SyntheticConverter(source_3, license="restricted").export_to_bonehub_format(
            cls.root, 3, num_workers=1, verbose=False
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_1_matching_finds_the_shared_image_only(self):
        found, errors = find_identical_images(self.root, workers=1, verbose=False)
        self.assertEqual(errors, {})
        self.assertEqual([link.scans for link in found], [("ds-001/sub-000001/scan-01", "ds-002/sub-000001/scan-01")])
        self.assertGreaterEqual(found[0].score, 5)
        self.assertEqual(load_links(self.root), found)
        # a decision made by hand survives finding the link again
        save_links(self.root, [found[0].model_copy(update={"decision": "rejected"})])
        find_identical_images(self.root, workers=1, verbose=False)
        self.assertEqual(load_links(self.root)[0].decision, "rejected")
        save_links(self.root, found)

    def test_2_groups(self):
        keys = ["ds-001/sub-000001/scan-01", "ds-001/sub-000002/scan-01", "ds-002/sub-000001/scan-01"]
        links = [Link(scans=(keys[2], keys[0]), relation="same_image", evidence="test", decision="confirmed")]
        image_group, subject_group = scan_groups(keys, links)
        self.assertEqual(image_group[keys[2]], keys[0])
        self.assertEqual(subject_group[keys[2]], "ds-001/sub-000001")
        self.assertEqual(image_group[keys[1]], keys[1])
        pending = [links[0].model_copy(update={"decision": "pending"})]
        self.assertEqual(scan_groups(keys, pending)[0][keys[2]], keys[2])

    def test_3_index(self):
        find_identical_images(self.root, workers=1, verbose=False)
        paths = build_index(self.root)
        scans = pq.read_table(paths["scans"]).to_pylist()
        self.assertEqual(len(scans), 4)
        first = scans[0]
        self.assertEqual(first["scan"], "ds-001/sub-000001/scan-01")
        self.assertEqual(first["image_size"], [16, 16, 12])
        self.assertEqual(first["image_spacing"], [0.8, 0.8, 2.0])
        self.assertEqual(first["segmentation"], ["FEMUR_LEFT", "FEMUR_RIGHT"])
        self.assertEqual(first["truncated"], ["FEMUR_LEFT"])
        self.assertEqual(first["gender"], "F")
        shared = [row for row in scans if row["image_group"] == "ds-001/sub-000001/scan-01"]
        self.assertEqual(len(shared), 2)
        self.assertEqual({row["subject_group"] for row in shared}, {"ds-001/sub-000001"})
        restricted = [row for row in scans if row["restricted"]]
        self.assertEqual([row["dataset_id"] for row in restricted], [3])
        labels = pq.read_table(paths["labels"]).to_pylist()
        femur = [row for row in labels if row["scan"] == first["scan"] and row["label"] == "FEMUR_LEFT"]
        self.assertEqual({(row["kind"], row["origin"], row["truncated"]) for row in femur},
                         {("segmentation", "source", True), ("mesh", "derived:test-mesher@1.0", None)})
        datasets = pq.read_table(paths["datasets"]).to_pylist()
        self.assertEqual([(row["dataset_id"], row["subjects"], row["scans"]) for row in datasets], [(1, 2, 2), (2, 1, 1), (3, 1, 1)])

    def test_4_hub_files_and_webpage(self):
        build_index(self.root)
        write_hub_files(self.root, "Org/Test-Dataset", "Test Dataset", homepage="https://example.org")
        self.assertEqual(validate(self.root, publishing=True), [])
        card = (self.root / "README.md").read_text(encoding="utf-8")
        self.assertTrue(card.startswith("---\npretty_name: \"Test Dataset\"\nlicense: other\n"))
        self.assertIn("data_files: index/scans.parquet", card)
        self.assertIn("Dataset 3 (Synthetic) is under a restricted license", card)
        gitignore = (self.root / ".gitignore").read_text(encoding="utf-8")
        for line in ("upload.md", "logs/", "**/mesh/", "/ds-003/"):
            self.assertIn(f"\n{line}\n", gitignore)
        self.assertIn("python -m bonehub_format upload", (self.root / "upload.md").read_text(encoding="utf-8"))
        self.assertIn("restricted", (self.root / "LICENSE.md").read_text(encoding="utf-8").lower())

        patterns = [pattern for _, pattern in excluded_from_upload(self.root)]
        def uploaded(path):
            return not any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
        for path in ("upload.md", "logs/ds-001.log", "ds-001/sub-000001/scan-01/mesh/FEMUR_LEFT.stl", "ds-003/dataset.json",
                     "ds-003/sub-000001/scan-01/image.nii.gz", ".cache/fingerprints/x.npz"):
            self.assertFalse(uploaded(path), path)
        for path in ("README.md", "ds-001/sub-000001/scan-01/mesh.zip", "ds-001/sub-000001/scan-01/image.nii.gz",
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
