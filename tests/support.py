"""A small synthetic source dataset and a converter for it, shared by the tests.

The handlers are module-level functions so that the converter's worker processes can import them.
"""

import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from bonehub_format import BoneLabelMap as BLM
from bonehub_format import DatasetInfo, ScanInfo, SubjectInfo
from bonehub_format.converter import DatasetConverter, ScanSource, SubjectSource
from bonehub_format.export import export_nii_segmentation

SHAPE = (12, 16, 16)  # z, y, x
LABEL_MAPPING = {1: BLM.FEMUR_LEFT.value, 2: BLM.FEMUR_RIGHT.value}
FAIL_ON = "fail_me"  # a subject whose segmentation export raises


def ct_volume(seed: int) -> np.ndarray:
    """CT-like values in HU: integers, with noise so that every slice differs."""
    rng = np.random.default_rng(seed)
    volume = rng.normal(40, 30, SHAPE).round()
    volume[:, :, :3] = -1000 + rng.normal(0, 5, (SHAPE[0], SHAPE[1], 3)).round()
    return volume


def write_image(array: np.ndarray, path: Path, dtype=np.float64) -> None:
    image = sitk.GetImageFromArray(array.astype(dtype))
    image.SetSpacing((0.8, 0.8, 2.0))
    image.SetOrigin((-10.0, -20.0, 30.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def make_source(folder: Path, subjects: dict[str, list[str]], first_seed: int = 0) -> Path:
    """Write <folder>/<subject>/<scan>/ with image.nii.gz (float64), labels.nii.gz and FEMUR_LEFT.stl. The images are
    ct_volume(first_seed), ct_volume(first_seed + 1), ... in the order given."""
    seed = first_seed
    for subject, scans in subjects.items():
        for scan in scans:
            scan_folder = folder / subject / scan
            write_image(ct_volume(seed), scan_folder / "image.nii.gz")
            labels = np.zeros(SHAPE, dtype=np.uint8)
            labels[0:4, 4:8, 4:8] = 1  # touches the first slice: cut off by the field of view
            labels[5:9, 6:10, 9:13] = 2
            write_image(labels, scan_folder / "labels.nii.gz", dtype=np.uint8)
            (scan_folder / "FEMUR_LEFT.stl").write_bytes(b"solid FEMUR_LEFT\nendsolid FEMUR_LEFT\n")
            seed += 1
    return folder


def read_dataset(dataset_root: Path) -> list[SubjectSource]:
    subjects = []
    for subject_folder in dataset_root.iterdir():  # not sorted: the converter numbers them in a fixed order
        scans = [
            ScanSource(
                info=ScanInfo(source_scan_id=f"{subject_folder.name}/{scan_folder.name}", modality="CT", age=60),
                image_path=scan_folder / "image.nii.gz",
                segmentation_paths=[scan_folder / "labels.nii.gz"],
                mesh_paths=[scan_folder / "FEMUR_LEFT.stl"],
                mesh_origin="derived:test-mesher@1.0",
            )
            for scan_folder in subject_folder.iterdir()
        ]
        if len(scans) == 2:
            scans[1].registered_to = scans[0].info.source_scan_id
        subjects.append(SubjectSource(info=SubjectInfo(source_subject_id=subject_folder.name, gender="F"), scans=scans))
    return subjects


def export_segmentation(scan: ScanSource, output_path: Path) -> None:
    if FAIL_ON in scan.info.source_scan_id:
        raise RuntimeError("This subject fails on purpose.")
    export_nii_segmentation(scan.segmentation_paths, output_path, [LABEL_MAPPING])


def export_mesh(scan: ScanSource, output_folder: Path) -> None:
    for path in scan.mesh_paths:
        shutil.copyfile(path, output_folder / path.name)


class SyntheticConverter(DatasetConverter):
    def __init__(self, dataset_root: Path, license: str = "CC-BY-4.0"):
        info = DatasetInfo(name="Synthetic", license=license, annotated_labels=["FEMUR_RIGHT", "FEMUR_LEFT"])
        super().__init__(dataset_root, info)
        self.custom_data_handlers.read_dataset = read_dataset
        self.custom_data_handlers.export_segmentation = export_segmentation
        self.custom_data_handlers.export_mesh = export_mesh
