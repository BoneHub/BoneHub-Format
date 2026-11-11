# BoneHub-Format

The BoneHub format is a folder structure for 3D bone datasets: medical images, bone segmentations, and the meshes
and NURBS surfaces made from them, gathered from many source datasets under one label map. This package defines the
format and provides the tools to build, check and publish datasets in it:

- **schema**: the metadata files, the label map and the segmentation file format;
- **conversion**: a base class for converting a source dataset, and helpers for images, NIfTI and DICOM segmentations;
- **checking**: a validator, an index of every scan and label, and a matcher that finds the same image in several datasets;
- **publishing**: the files that make a folder a Hugging Face dataset repository, and a webpage for GitHub Pages.

The [BoneHub Dataset](https://bonehub.github.io/BoneHub-Dataset) is built with it.

## Install

```bash
pip install "bonehub-format @ git+https://github.com/BoneHub/BoneHub-Format.git"              # read and check datasets
pip install "bonehub-format[converter] @ git+https://github.com/BoneHub/BoneHub-Format.git"   # also build and publish them
pip install -e ".[converter]"                                                                 # from a local clone
```

The base install needs only pydantic: it reads, validates and packs datasets. The `io` extra adds reading and writing
segmentation masks (numpy, SimpleITK); the `converter` extra adds everything needed to convert sources and to build
the index, links, Hugging Face files and webpage.

## The format

```
<root>/
├── README.md, LICENSE.md, .gitattributes, .gitignore   the Hugging Face files (generated)
├── upload.md                                           how to upload the root (generated, not uploaded)
├── links.json                                          scans found to be the same image or person
├── index/                                              datasets / scans / labels .parquet (generated)
├── github-webpage/                                     the dataset's webpage (generated)
├── logs/                                               conversion logs (not uploaded)
└── ds-001/                                             a dataset: one source dataset
    ├── dataset.json
    └── sub-000001/                                     a subject: one person
        ├── subject.json
        └── scan-01/                                    a scan: one image of the subject
            ├── scan.json
            ├── image.nii.gz
            ├── segmentation.seg.nrrd
            ├── mesh/    <LABEL>.stl                    per-label files, kept locally
            ├── nurbs/   <LABEL>.iges, <LABEL>.step
            ├── mesh.zip                                the zips of those folders, which are published
            └── nurbs.zip
```

A folder's name is its ID; no file name or metadata file repeats it. A scan is identified everywhere by its path,
e.g. `ds-001/sub-000001/scan-01`. A dataset holds at most 9,999 subjects (Hugging Face allows 10,000 entries in a
folder) and a subject at most 99 scans.

### Metadata files

The metadata files hold only what cannot be read from the data files. Image size and spacing, label counts and which
bones touch the image border are computed into the index.

**dataset.json**: `name`, `description`, `url`, `paper`, `citation` (BibTeX), `country`, `release_date`, `version` (of
the source), `license`, `derived_from`, `annotated_labels`, `remarks`, `schema_version`.

**subject.json**: `source_subject_id`, `gender` (`M`, `F`, `O`; absent when unknown), `remarks`.

**scan.json**: `source_scan_id`, `modality` (a DICOM code: `CT`, `MR`, ...), `age` (years), `weight` (kg), `height` (cm),
`bmi`, `postmortem`, `days_from_baseline`, `registered_to`, `source_split`, `license`, `segmentation`, `mesh`, `nurbs`,
`remarks`. Everything that can change between scans of a person is here, not in subject.json.

```json
{
    "source_scan_id": "train/Task1/pelvis/1PA001/ct.nii.gz",
    "modality": "CT",
    "registered_to": "scan-02",
    "source_split": "train",
    "segmentation": {"SACRUM": "source", "HIP_BONE_LEFT": "model:TotalSegmentator@2.4.0"},
    "mesh": {"SACRUM": "derived:bonehub-mesher@1.0"},
    "nurbs": {"SACRUM": "derived:mesh2nurbs-rhino3d@1.0"}
}
```

- **Labels and origins.** `segmentation`, `mesh` and `nurbs` map each label that has data of that kind to its origin:
  `source` (it came with the source dataset), `manual` (drawn by hand for this dataset), `model:<name>@<version>`, or
  `derived:<tool>@<version>` (computed from another file of the same scan, e.g. a mesh from the segmentation). A label
  that is not listed has no data of that kind.
- **Coverage.** `annotated_labels` lists the labels a dataset's segmentations cover. In a scan with a segmentation, a
  covered label that is missing is absent from the image; a label that is not covered was not segmented. An empty
  list means nothing is segmented; no list means it is unknown or differs between scans.
- **Licenses.** `license` is an SPDX identifier (`CC-BY-4.0`, `CC-BY-NC-SA-4.0`, ... see
  `bonehub_format.licenses.LICENSES`) or `restricted`, for data under a custom license with restricted access. A
  scan's `license` overrides its dataset's. Restricted data is converted like any other but is not published.
- **Derived datasets.** A dataset made from another one, such as a quality-checked copy, gets its own ID and names its
  parents in `derived_from`; the `source_scan_id` of each of its scans is the path of the scan it was made from.
  Remarks about the quality check go in its dataset.json.
- **Registration.** `registered_to` names the scan of the same subject an image was registered to.

### Data files

- **Images** are NIfTI, on the grid of the source image (never resampled by the conversion). CT is stored as int16 when
  that is lossless, which it is for HU values; float64 images are stored as float32.
- **Segmentations** are 3D Slicer segmentation files (`.seg.nrrd`), on the grid of the image; see below.
- **Meshes** (`.stl`) and **NURBS surfaces** (`.iges`, `.step`) are in the physical space of the image, in LPS and
  millimetres, one file per label and file type. A scan keeps them in `mesh/` and `nurbs/`; the zips of these folders
  are what gets published.

### Rules

- **IDs are reproducible.** Subjects are numbered in the natural order of their `source_subject_id` (`case_2` before
  `case_10`), and the scans of a subject by `days_from_baseline` and then `source_scan_id`. Converting the same source
  again gives the same IDs.
- **A dataset does not change once converted.** Fixing a converter means converting the dataset again under the same
  ID; curating it (e.g. a quality check) or converting a new release of its source gives a new dataset.
- **Versions.** `schema_version` in dataset.json is the version of this format (major.minor). A minor version adds
  something datasets of older minor versions still satisfy, such as a new optional field or a new kind of per-label
  data, which gets a new file or zip in the scan folder and a new label-to-origin block in scan.json. A major version
  makes older datasets invalid. A package reads datasets of its own major version and an equal or lower minor version.

### Matching scans across datasets

Source datasets reuse each other's images (CTPelvic1K, for one, includes KiTS19 and CT Colonography images), so the
same image or person can appear in several datasets. `links.json` records such matches:

```json
{"links": [{"scans": ["ds-002/sub-000012/scan-01", "ds-007/sub-000300/scan-01"],
            "relation": "same_image", "evidence": "identical_slices", "score": 87,
            "decision": "confirmed", "note": null}]}
```

`same_image` means the same acquisition, possibly re-saved, reoriented or cropped; `same_subject` means the same
person. Only confirmed links count; a rejected link is kept so that it is not proposed again. The links of a derived
dataset to its parent are declared by its metadata. The index joins linked scans into `image_group` and
`subject_group`: drop duplicates by `image_group`, and split training and test data by `subject_group`.

`python -m bonehub_format match ROOT` finds the scans that are the same image. It fingerprints every slice of every
image, along each array axis, by the histogram of its values after subtracting the slice's minimum; a histogram does
not change with rotation, flipping, transposition or a constant intensity offset, so a match does not depend on the
file format, pixel type or header orientation. Slices with little information (air, padding) are skipped. Two scans
are the same image when at least 5 slices match in an order a crop, flip or shift explains; real image noise makes
chance matches of whole slices practically impossible, so these links are confirmed without review. Resampled or
intensity-clipped copies have no identical slices and are not found.

### The index

`index/` holds three Parquet tables, rebuilt from everything above by `python -m bonehub_format index ROOT`:

- `datasets.parquet`: one row per dataset, with its metadata and its number of subjects and scans;
- `scans.parquet`: one row per scan, with the fields of its subject, its effective license, the image size and
  spacing, its labels per kind, the segmented labels that touch the image border (`truncated`), and its
  `image_group` and `subject_group`;
- `labels.parquet`: one row per scan, label and kind, with the origin.

```python
import pandas as pd

scans = pd.read_parquet("Z:/BoneHub/BoneHub_Dataset/index/scans.parquet")
femurs = scans[scans["segmentation"].apply(lambda labels: "FEMUR_LEFT" in labels)]
```

## Reading a dataset

```python
from bonehub_format import Root

root = Root("Z:/BoneHub/BoneHub_Dataset")
for scan in root.scans():
    print(scan.key, scan.info.modality, scan.subject.info.gender, scan.license)
    print(scan.image_path, scan.segmentation_path, scan.info.labels("segmentation"))
    for label, files in scan.model_files("mesh").items():  # from mesh/, or from mesh.zip
        stl_bytes = files[0].read_bytes()
```

## Converting a dataset

A converter subclasses `DatasetConverter` and sets its handlers. `read_dataset` lists the subjects and their scans;
the image is read from `image_path` (a file or a DICOM folder) unless `export_image` is set:

```python
from pathlib import Path

from bonehub_format import BoneLabelMap as BLM, DatasetInfo, ScanInfo, SubjectInfo
from bonehub_format.converter import DatasetConverter, ScanSource, SubjectSource
from bonehub_format.export import export_nii_segmentation

LABELS = {1: BLM.FEMUR_LEFT.value, 2: BLM.FEMUR_RIGHT.value}


def read_dataset(root: Path) -> list[SubjectSource]:
    return [
        SubjectSource(
            info=SubjectInfo(source_subject_id=case.name, gender="F"),
            scans=[
                ScanSource(
                    info=ScanInfo(source_scan_id=case.name, modality="CT", age=63),
                    image_path=case / "ct.nii.gz",
                    segmentation_paths=[case / "labels.nii.gz"],
                )
            ],
        )
        for case in sorted(root.iterdir())
    ]


def export_segmentation(scan: ScanSource, output_path: Path):
    export_nii_segmentation(scan.segmentation_paths, output_path, [LABELS])


class MyDataset(DatasetConverter):
    def __init__(self, dataset_root: Path):
        info = DatasetInfo(name="My dataset", license="CC-BY-4.0", annotated_labels=["FEMUR_LEFT", "FEMUR_RIGHT"])
        super().__init__(dataset_root, info)
        self.custom_data_handlers.read_dataset = read_dataset
        self.custom_data_handlers.export_segmentation = export_segmentation


if __name__ == "__main__":  # subjects are converted in worker processes, which re-import this script
    MyDataset(Path("path/to/source")).export_to_bonehub_format(Path("path/to/root"), output_dataset_id=1)
```

A subject's folder is written completely or not at all, and `subject.json` is written last. `overwrite=True`
converts a dataset again from scratch; `resume=True` keeps the subjects an earlier run finished and converts the rest,
after checking that the source still lists the same subjects. Logs go to `<root>/logs/`.

The converters of the BoneHub Dataset are in the private `custom_dataset_io` submodule.

## Publishing

```bash
python -m bonehub_format prepare ROOT --repo-id BoneHub/BoneHub-Dataset --title "BoneHub Dataset" --homepage https://bonehub.github.io/BoneHub-Dataset
python -m bonehub_format upload ROOT --repo-id BoneHub/BoneHub-Dataset
```

`prepare` brings the zips up to date, rebuilds the index, validates the root, and writes:

- `README.md`, the dataset card, which the Hugging Face dataset viewer shows with the index tables;
- `LICENSE.md`, the license of every dataset and of the scans with their own;
- `.gitattributes` and `.gitignore`;
- `upload.md`, the steps to upload (it is not uploaded itself);
- `github-webpage/`, the dataset's webpage, with the steps to publish it on GitHub Pages in its README. Pass
  `--webpage-options options.json` to set its subtitle, About and Contact sections, and analytics snippet (see
  `bonehub_format.webpage.build_webpage`).

`upload` sends the root to Hugging Face without `upload.md`, the logs and caches, the per-label `mesh/` and `nurbs/`
folders (their zips are uploaded) and restricted data. Use it rather than `hf upload`, which does not read
`.gitignore`.

Other commands: `validate ROOT [--publishing]`, `pack ROOT`, `unpack ROOT` (extract the zips of a downloaded copy into
folders), `index ROOT`, `match ROOT`.

## Segmentation files

Masks are stored as `.seg.nrrd`, 3D Slicer's segmentation format. Voxels hold per-file segment numbers (1, 2, 3 ...),
and the header maps each number to its `BoneLabelMap` label and records its colour and bounding box. Open them in 3D
Slicer as a **Segmentation**; in Python, `read_segmentation` returns the mask as `BoneLabelMap` values:

```python
import SimpleITK as sitk, numpy as np
from bonehub_format import BoneLabelMap, read_segmentation  # needs the [io] extra

mask = read_segmentation("segmentation.seg.nrrd")  # SimpleITK image of BoneLabelMap values
arr = sitk.GetArrayViewFromImage(mask)
print([BoneLabelMap(v).name for v in np.unique(arr) if v])
```

The values are int32; do not cast them to float32, which cannot hold 9-digit values exactly.
`read_segmentation_labels` lists the labels of a mask from its header alone, without numpy or SimpleITK.

## Label map

The label value is built from four independent fields, so a bone, a sub-part, a tissue compartment and a side each
occupy their own digits:

```
value = structure * 100000 + part * 1000 + tissue * 10 + side

FEMUR_PROXIMAL_CORTICAL_LEFT = 710001011  =  7100 | 01 | 01 | 1
                                              |      |    |    +-- side       1 = left
                                              |      |    +------- tissue    01 = cortical
                                              |      +------------ part      01 = proximal
                                              +------------------- structure 7100 = femur
```

Every value has 9 digits (4 + 2 + 2 + 1), except `BACKGROUND = 0`. Use `structure_of`, `part_of`, `tissue_of` and
`side_of` to group labels — e.g. every left-femur label is `structure_of(v) == 7100 and side_of(v) == 1`.

## Related repositories

| Repository | Purpose |
| --- | --- |
| [bonehub_dataset_quality_check_server](https://github.com/BoneHub/bonehub_dataset_quality_check_server) | Dockerised server that distributes subjects to reviewers and collects their corrected segmentations (not yet updated to this version of the format) |
| [bonehub_dataset_quality_check_3dslicer_extension](https://github.com/BoneHub/bonehub_dataset_quality_check_3dslicer_extension) | 3D Slicer extension reviewers use to fetch, correct and submit segmentations |

## License
MIT
