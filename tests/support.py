"""A small synthetic source dataset and a converter for it, shared by the tests.

The handlers are module-level functions so that the converter's worker processes can import them.
"""

import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from bonehub_format import LabelMap as LM
from bonehub_format import Licenses, ScanInfo, SourceInfo, SubjectInfo
from bonehub_format.converter import Contribution, CustomDataHandlers, DatasetConverter, ScanSource, SubjectSource
from bonehub_format.export import export_nii_segmentation

SHAPE = (12, 16, 16)  # z, y, x
LABEL_MAPPING = {1: LM.FEMUR_LEFT, 2: LM.FEMUR_RIGHT}
SOURCE_ID = 3  # the synthetic source's BoneHub ID
PROVIDER_ID = 7  # a second synthetic source, which gives every scan of the first a segmentation of its own
PROVIDER_MAPPING = {1: LM.FEMUR_LEFT, 2: LM.GLUTEUS_MAXIMUS_RIGHT}
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
                source_scan_id=f"{subject_folder.name}/{scan_folder.name}",
                info=ScanInfo(modality="CT", age=60),
                image_path=scan_folder / "image.nii.gz",
                segmentation_paths=[scan_folder / "labels.nii.gz"],
                mesh_paths=[scan_folder / "FEMUR_LEFT.stl"],
                mesh_origin="derived:test-mesher@1.0",
            )
            for scan_folder in subject_folder.iterdir()
        ]
        if len(scans) == 2:
            scans[1].registered_to = scans[0].source_scan_id
        subjects.append(SubjectSource(source_subject_id=subject_folder.name, info=SubjectInfo(gender="F"), scans=scans))
    return subjects


def read_dataset_with_contributions(dataset_root: Path) -> list[SubjectSource]:
    """read_dataset, with the segmentation source PROVIDER_ID gives each scan: the same label file, other labels."""
    subjects = read_dataset(dataset_root)
    for subject in subjects:
        subject.other_sources = {PROVIDER_ID: f"provider-{subject.source_subject_id}"}
        for scan in subject.scans:
            scan.contributions = [
                Contribution(
                    source_id=PROVIDER_ID,
                    source_scan_id=f"provider-{scan.source_scan_id}",
                    segmentation_paths=scan.segmentation_paths,
                    segmentation_origin="model:TestModel@2.0",
                    handlers=CustomDataHandlers(export_segmentation=export_provider_segmentation),
                )
            ]
    return subjects


def read_dataset_with_moved_contribution(dataset_root: Path) -> list[SubjectSource]:
    """read_dataset_with_contributions, but the provider's labels are labels_moved.nii.gz, which the test writes on
    another grid (shifted, flipped ...)."""
    subjects = read_dataset_with_contributions(dataset_root)
    for subject in subjects:
        for scan in subject.scans:
            scan.contributions[0].segmentation_paths = [path.with_name("labels_moved.nii.gz") for path in scan.segmentation_paths]
    return subjects


def export_provider_segmentation(scan: Contribution, output_path: Path) -> None:
    export_nii_segmentation(scan.segmentation_paths, output_path, [PROVIDER_MAPPING])


def export_segmentation(scan: ScanSource, output_path: Path) -> None:
    if FAIL_ON in scan.source_scan_id:
        raise RuntimeError("This subject fails on purpose.")
    export_nii_segmentation(scan.segmentation_paths, output_path, [LABEL_MAPPING])


def export_mesh(scan: ScanSource, output_folder: Path) -> None:
    for path in scan.mesh_paths:
        shutil.copyfile(path, output_folder / path.name)


#: The landmarks every scan of the landmark converter gets: a sphere, two raters' points, an axis, an area patch.
LANDMARKS = [
    {"name": "LANDMARK_FEMUR_HEAD_CENTER_LEFT", "type": "sphere", "center": [4.0, 5.0, 2.0], "radius": 1.5,
     "source_name": "femur head"},
    {"name": "LANDMARK_FEMUR_GREATER_TROCHANTER_LEFT", "type": "point", "position": [7.0, 6.5, 3.0], "rater": 1},
    {"name": "LANDMARK_FEMUR_GREATER_TROCHANTER_LEFT", "type": "point", "position": [7.2, 6.4, 3.1], "rater": 2},
    {"name": "LANDMARK_KNEE_JOINT_FLEXION_AXIS_LEFT", "type": "line", "points": [[1.0, 1.0, 9.0], [9.0, 1.0, 9.0]]},
    {"name": "LANDMARK_GLUTEUS_MEDIUS_ORIGIN_LEFT", "type": "area", "file": "LANDMARK_GLUTEUS_MEDIUS_ORIGIN_LEFT.stl",
     "element": "anterior"},
]


def read_dataset_with_landmarks(dataset_root: Path) -> list[SubjectSource]:
    """read_dataset, with landmarks on every scan (the same mesh file stands in for the area patch)."""
    subjects = read_dataset(dataset_root)
    for subject in subjects:
        for scan in subject.scans:
            scan.landmarks_paths = scan.mesh_paths
            scan.landmarks_origin = "manual"
    return subjects


def export_landmarks(scan: ScanSource, output_folder: Path) -> None:
    from bonehub_format.landmarks import write_landmarks

    shutil.copyfile(scan.landmarks_paths[0], output_folder / "LANDMARK_GLUTEUS_MEDIUS_ORIGIN_LEFT.stl")
    write_landmarks(output_folder, LANDMARKS)


class SyntheticConverter(DatasetConverter):
    """The synthetic source SOURCE_ID; with contributions, its scans also hold the segmentations of PROVIDER_ID."""

    def __init__(
        self,
        dataset_root: Path,
        license: str | Licenses = "CC-BY-4.0",
        contributions: bool = False,
        provider_license: str | Licenses = "CC-BY-NC-4.0",
        source_id: int = SOURCE_ID,
        landmarks: bool = False,
    ):
        licenses = license if isinstance(license, Licenses) else Licenses.all(license)
        source = SourceInfo(name="Synthetic", license=licenses, annotated_labels=["FEMUR_RIGHT", "FEMUR_LEFT"])
        provider_licenses = provider_license if isinstance(provider_license, Licenses) else Licenses.all(provider_license)
        others = {PROVIDER_ID: SourceInfo(name="Synthetic provider", license=provider_licenses)} if contributions else None
        super().__init__(dataset_root, source_id, source, other_sources=others)
        self.custom_data_handlers.read_dataset = read_dataset_with_contributions if contributions else read_dataset
        self.custom_data_handlers.export_segmentation = export_segmentation
        self.custom_data_handlers.export_mesh = export_mesh
        if landmarks:
            self.custom_data_handlers.read_dataset = read_dataset_with_landmarks
            self.custom_data_handlers.export_landmarks = export_landmarks
