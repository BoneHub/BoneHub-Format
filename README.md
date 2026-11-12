# BoneHub-Format

The BoneHub format is a folder structure for harmonized databases of 3D bone data: medical images, bone
segmentations, and the meshes and NURBS surfaces made from them, gathered from many source datasets under one label
map. This package defines the format and provides the tools to build, check and publish databases in it:

- **schema**: the metadata files, the label map and the segmentation file format;
- **conversion**: a base class for converting a source dataset, and helpers for images, NIfTI and DICOM segmentations;
- **checking**: a validator, which also checks dataset IDs against the BoneHub registry, and an index of every scan
  and label;
- **changing**: helpers that add, replace and remove the labels of a converted scan;
- **publishing**: the files that make a folder a HuggingFace dataset repository, and a webpage for GitHub Pages.

The databases built with it are published in the BoneHub [Harmonized Databases](https://huggingface.co/collections/BoneHub/harmonized-databases) collection on
HuggingFace.

## Install

```bash
pip install "bonehub-format @ git+https://github.com/BoneHub/BoneHub-Format.git"              # read and check databases
pip install "bonehub-format[converter] @ git+https://github.com/BoneHub/BoneHub-Format.git"   # also build and publish them
pip install -e ".[converter]"                                                                 # from a local clone
```

The base install needs only pydantic: it reads, validates and packs databases. The `io` extra adds reading and writing
segmentation masks (numpy, SimpleITK); the `converter` extra adds everything needed to convert sources and to build
the index, HuggingFace files and webpage.

## The format

A **database** is a root folder of datasets harmonized under one label map, published as one HuggingFace dataset
repository. Each database has its own scope (for example only whole-body scans, or several body regions together)
and its own index and webpage. A **dataset** is one source dataset in a database.

```
<database>/
├── README.md, LICENSE.md, .gitattributes, .gitignore   the HuggingFace files (generated)
├── upload.md                                           how to upload the database (generated, not uploaded)
├── index/                                              datasets / scans / labels .parquet (generated)
├── github-webpage/                                     the database's webpage (generated)
├── logs/                                               conversion logs (not uploaded)
└── ds-0027/                                            a dataset, named after its BoneHub ID
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

A folder's name is its ID; no file name or metadata file repeats it. A scan is identified by its path, e.g.
`ds-0027/sub-000001/scan-01`. A subject holds at most 99 scans.

### Dataset IDs

A dataset's ID is its **BoneHub ID** in the
[BoneHub registry](https://github.com/BoneHub/Public-Datasets/blob/main/data/bonehub_public_datasets.csv), written
with four digits: TotalSegmentator CT, BoneHub ID 27, is `ds-0027` in every database that holds it. The registry is
the only source of dataset IDs: `validate`, `prepare` and `upload` check every dataset folder and every `derived_from`
against it (`bonehub_format.load_registry` reads it).

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
- **Derived datasets.** `derived_from` lists the BoneHub IDs of the datasets a dataset was made from, wholly or in
  part. It is the only record of which datasets hold the same people: BoneHub does not try to recognise the same
  person or image across datasets. Where a scan was made from a scan of a BoneHub-format dataset, its `source_scan_id`
  is that scan's path, e.g. `ds-0027/sub-000001/scan-01`.
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
- **Changing a dataset is your choice.** A change such as segmenting another bone, correcting a mask or adding meshes
  can be made **in place**: the dataset keeps its ID, and the origin of each label in scan.json tells the new labels
  from the source's (see [Changing a dataset](#changing-a-dataset)). Or it can be published as a **new dataset**: add
  a row for it to the registry, convert or copy it under its new BoneHub ID, and name the original in `derived_from`.
  Fixing a converter means converting the dataset again under the same ID; this replaces the dataset folder, so
  changes made in place since then must be made again.
- **Versions.** `schema_version` in dataset.json is the version of this format (major.minor). A minor version adds
  something datasets of older minor versions still satisfy, such as a new optional field or a new kind of per-label
  data, which gets a new file or zip in the scan folder and a new label-to-origin block in scan.json. A major version
  makes older datasets invalid. A package reads datasets of its own major version and an equal or lower minor version.

### The index

`index/` holds three Parquet tables, rebuilt from everything above by `python -m bonehub_format index DATABASE`:

- `datasets.parquet`: one row per dataset, with its metadata and its number of subjects and scans;
- `scans.parquet`: one row per scan, with the fields of its subject, its effective license, the image size and
  spacing, its labels per kind, and the segmented labels that touch the image border (`truncated`);
- `labels.parquet`: one row per scan, label and kind, with the origin.

```python
import pandas as pd

scans = pd.read_parquet("path/to/database/index/scans.parquet")
femurs = scans[scans["segmentation"].apply(lambda labels: "FEMUR_LEFT" in labels)]
```

## Reading a database

```python
from bonehub_format import Database

database = Database("path/to/database")
for scan in database.scans():
    print(scan.key, scan.info.modality, scan.subject.info.gender, scan.license)
    print(scan.image_path, scan.segmentation_path, scan.info.labels("segmentation"))
    for label, files in scan.model_files("mesh").items():  # from mesh/, or from mesh.zip
        stl_bytes = files[0].read_bytes()
```

## Changing a dataset

To change a dataset in place, `update_segmentation` and `update_model_files` add, replace and remove the labels of a
scan and keep its scan.json in step with the files:

```python
from bonehub_format import Database, update_model_files, update_segmentation

scan = Database("path/to/database").dataset(27).subject(1).scan(1)

# Add or replace the labels a model segmented. The mask holds BoneLabelMap values on the grid of the scan's image;
# where it overlaps another label, overlap="replace" (the default) or "keep" decides which keeps the voxel.
update_segmentation(scan, "patella_mask.nii.gz", origin="model:MyModel@1.0")
update_segmentation(scan, remove=["PATELLA_LEFT"])

update_model_files(scan, "mesh", {"PATELLA_LEFT": Path("patella.stl")}, origin="derived:my-mesher@1.0")
```

Neither changes dataset.json: when a label is now segmented in every scan, add it to `annotated_labels`. Rebuild the
index afterwards. `update_segmentation` needs the `io` extra.

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
    MyDataset(Path("path/to/source")).export_to_bonehub_format(Path("path/to/database"), output_dataset_id=27)
```

A subject's folder is written completely or not at all, and `subject.json` is written last. `overwrite=True`
converts a dataset again from scratch; `resume=True` keeps the subjects an earlier run finished and converts the rest,
after checking that the source still lists the same subjects. Logs go to `<database>/logs/`.

The converters of the BoneHub source datasets are in the private `custom_dataset_io` submodule.

## Publishing

```bash
python -m bonehub_format prepare DATABASE --repo-id BoneHub/<database> --title "<title>" --homepage https://bonehub.github.io/<database>
python -m bonehub_format upload DATABASE --repo-id BoneHub/<database>
```

`prepare` brings the zips up to date, rebuilds the index, validates the database, and writes:

- `README.md`, the dataset card, which the HuggingFace dataset viewer shows with the index tables;
- `LICENSE.md`, the license of every dataset and of the scans with their own;
- `.gitattributes` and `.gitignore`;
- `upload.md`, the steps to upload (it is not uploaded itself);
- `github-webpage/`, the database's webpage, with the steps to publish it on GitHub Pages in its README. Pass
  `--webpage-options options.json` to set its subtitle, About and Contact sections, and analytics snippet (see
  `bonehub_format.webpage.build_webpage`).

`upload` sends the database to HuggingFace without `upload.md`, the logs, the per-label `mesh/` and `nurbs/`
folders (their zips are uploaded) and restricted data. Use it rather than `hf upload`, which does not read
`.gitignore`. Once the repository is public, add it to the [Harmonized Databases](https://huggingface.co/collections/BoneHub/harmonized-databases) collection.

Other commands: `validate DATABASE [--publishing]`, `pack DATABASE`, `unpack DATABASE` (extract the zips of a
downloaded copy into folders), `index DATABASE`. `validate`, `prepare` and `upload` read the registry from GitHub;
pass `--registry path/to/registry.csv` to use a local copy, or `validate --registry none` to skip the ID check.

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
