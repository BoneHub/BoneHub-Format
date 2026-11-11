"""Convert the first subjects of each source dataset with the private converters, then read and validate them.

Needs the private custom_dataset_io submodule and the source datasets in PUBLIC_DATASETS; skipped without them.
"""

import os

os.environ["MAX_SUBJECTS_FOR_TESTING"] = "2"  # Set a limit for testing purposes

import unittest
from pathlib import Path

from bonehub_format import Root, validate

try:
    import bonehub_format.custom_dataset_io as custom_dataset_io
except ImportError:
    custom_dataset_io = None

# custom_dataset_io is a private git submodule; without access its folder is empty.
if not hasattr(custom_dataset_io, "__all__"):
    raise unittest.SkipTest("custom_dataset_io submodule is not available.")

PUBLIC_DATASETS = Path("Z:/BoneHub/Public_Datasets")
if not PUBLIC_DATASETS.is_dir():
    raise unittest.SkipTest(f"The source datasets are not available in {PUBLIC_DATASETS}.")

OUTPUT_ROOT = Path(__file__).resolve().parent / "dummy_bonehub_dataset"


class _ConvertAndRead:
    converter: str  # class name in custom_dataset_io
    source: str  # folder in PUBLIC_DATASETS
    dataset_id: int
    subjects = 2  # MAX_SUBJECTS_FOR_TESTING per list of subjects the converter reads

    def test_convert_and_read(self):
        converter = getattr(custom_dataset_io, self.converter)(PUBLIC_DATASETS / self.source)
        converter.export_to_bonehub_format(OUTPUT_ROOT, self.dataset_id, overwrite=True, verbose=False)

        dataset = Root(OUTPUT_ROOT).dataset(self.dataset_id)
        self.assertEqual(len(dataset), self.subjects)
        self.assertEqual([s.id for s in dataset.subjects()], list(range(1, self.subjects + 1)))
        for scan in dataset.scans():
            self.assertIsNotNone(scan.image_path, scan.key)
            self.assertIsNotNone(scan.license, scan.key)
        problems = [p for p in validate(OUTPUT_ROOT) if dataset.path.name in p]
        self.assertEqual(problems, [])


class TestBoneDat(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "BoneDat", "124 BoneDat/BoneDat", 1


class TestKiTS2023(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "KiTS2023", "073 kits23", 2


class TestSpineMetsCTSeg(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "SpineMetsCTSeg", "019 TCIA Spine-Mets-CT-SEG", 3


class TestEnhancePET(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "EnhancePET", "134 enhance-pet-1_6k", 4


class TestVSDReconstruction(_ConvertAndRead, unittest.TestCase):
    converter, dataset_id = "VSDReconstruction", 5
    source = "036 VSDFullBodyBoneReconstruction/Hamid_processed/vsd-lower-extremities-seg"


class TestTotalSegmentatorCT(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "TotalSegmentatorCT", "027 Totalsegmentator/raw_data", 6


class TestCTPelvic1K(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "CTPelvic1K", "044 CTPelvic1K", 7


class TestPENGWIN(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "PENGWIN", "106 PENGWIN", 8


class TestCTPEL(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "CTPEL", "138 ctpel", 9


class TestSynthRAD2023(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "SynthRAD2023", "006 SynthRAD2023", 10
    subjects = 6  # 2 each of Task1 train (CT + MR), Task1 validation (MR) and Task2 (CT)


class TestVSDFullBody(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "VSDFullBody", "071 VSDFullBody", 11


class TestACRIN6664(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "ACRIN6664", "070 TCIA CT COLONOGRAPHY ACRIN 6664", 12


if __name__ == "__main__":
    unittest.main()
