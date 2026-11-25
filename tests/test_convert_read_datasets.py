"""Convert the first subjects of each source dataset with the private converters, then read and validate them.

Needs the private custom_dataset_io submodule and the source datasets in PUBLIC_DATASETS; skipped without them.
"""

import os

os.environ["MAX_SUBJECTS_FOR_TESTING"] = "2"  # Set a limit for testing purposes

import unittest
from pathlib import Path

from bonehub_format import Database, validate

try:
    import bonehub_format.custom_dataset_io as custom_dataset_io
except ImportError:
    custom_dataset_io = None

# custom_dataset_io is a private git submodule; without access its folder is empty.
if not hasattr(custom_dataset_io, "__all__"):
    raise unittest.SkipTest("custom_dataset_io submodule is not available.")

PUBLIC_DATASETS = Path("/mnt/nas/BoneHub/Public_Datasets")
if not PUBLIC_DATASETS.is_dir():
    raise unittest.SkipTest(f"The source datasets are not available in {PUBLIC_DATASETS}.")

OUTPUT_ROOT = Path(__file__).resolve().parent / "dummy_bonehub_dataset"


class _ConvertAndRead:
    converter: str  # class name in custom_dataset_io
    source: str  # folder in PUBLIC_DATASETS
    dataset_id: int  # its BoneHub ID
    subjects = 2  # MAX_SUBJECTS_FOR_TESTING per list of subjects the converter reads
    # Datasets whose data the converted scans hold (provider ID, converter, source): converted first, so that their
    # dataset.json is in the database. Those that only give data to other datasets write nothing else.
    providers: tuple[tuple[int, str, str], ...] = ()

    def test_convert_and_read(self):
        for dataset_id, converter, source in self.providers:
            getattr(custom_dataset_io, converter)(PUBLIC_DATASETS / source).export_to_bonehub_format(
                OUTPUT_ROOT, dataset_id, overwrite=True, verbose=False
            )
        converter = getattr(custom_dataset_io, self.converter)(PUBLIC_DATASETS / self.source)
        converter.export_to_bonehub_format(OUTPUT_ROOT, self.dataset_id, overwrite=True, verbose=False)

        dataset = Database(OUTPUT_ROOT).dataset(self.dataset_id)
        self.assertEqual(len(dataset), self.subjects)
        self.assertEqual([s.id for s in dataset.subjects()], list(range(1, self.subjects + 1)))
        for scan in dataset.scans():
            if scan.image_path is not None:
                self.assertIsNotNone(scan.license("image"), scan.key)
            self.assertTrue(scan.provider_ids, scan.key)
        names = [dataset.path.name] + [f"ds-{dataset_id:04d}" for dataset_id, _, _ in self.providers]
        problems = [p for p in validate(OUTPUT_ROOT) if any(name in p for name in names)]
        self.assertEqual(problems, [])


class TestBoneDat(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "BoneDat", "0124 BoneDat/BoneDat", 124


class TestKiTS2023(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "KiTS2023", "0073 kits23", 73


class TestSpineMetsCTSeg(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "SpineMetsCTSeg", "0019 TCIA Spine-Mets-CT-SEG", 19


class TestEnhancePET(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "EnhancePET", "0134 enhance-pet-1_6k", 134
    providers = (
        (18, "Atlas", "0018 Atlas dataset"),
        (40, "FDGPETCTLesions", "0040 TCIA FDG-PET-CT-Lesions"),
        (75, "PSMAFDGPETCTLesions", "0075 psma-fdg-pet-ct-lesions"),
    )


class TestPSMAFDGPETCTLesions(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "PSMAFDGPETCTLesions", "0075 psma-fdg-pet-ct-lesions", 75


class TestTotalSegmentatorCT(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "TotalSegmentatorCT", "0027 Totalsegmentator/v3.0.0", 27


class TestTotalSegmentatorMRI(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "TotalSegmentatorMRI", "0028 TotalsegmentatorMRI/v3.0.0", 28


class TestHealthyTotalBodyCTs(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "HealthyTotalBodyCTs", "0029 TCIA Healthy-Total-Body-CTs", 29


class TestQuadraHC(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "QuadraHC", "0163 QUADRA_HC", 163


class TestOpenSkeleton(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "OpenSkeleton", "0176 OpenSkeleton", 176


class TestCTPelvic1K(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "CTPelvic1K", "0044 CTPelvic1K", 44


class TestPENGWIN(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "PENGWIN", "0106 PENGWIN", 106


class TestCTPEL(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "CTPEL", "0138 ctpel", 138


class TestSynthRAD2023(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "SynthRAD2023", "0006 SynthRAD2023", 6
    subjects = 6  # 2 each of Task1 train (CT + MR), Task1 validation (MR) and Task2 (CT)


class TestVSDFullBody(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "VSDFullBody", "0071 VSDFullBody", 71
    providers = ((180, "VSDLowerExtremitiesSeg", "0180 vsd-lower-extremities-seg"),)


class TestVisibleHuman3DModels(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "VisibleHuman3DModels", "0153 visible-human-3d-models", 153
    providers = ((109, "VisibleHumanCOB", "0109 VisibleHuman"),)


class TestACRIN6664(_ConvertAndRead, unittest.TestCase):
    converter, source, dataset_id = "ACRIN6664", "0070 TCIA CT COLONOGRAPHY ACRIN 6664", 70


if __name__ == "__main__":
    unittest.main()
