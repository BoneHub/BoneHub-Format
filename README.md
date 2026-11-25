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
├── ds-0018/                                            a dataset whose data all lives in other datasets' scans
│   └── dataset.json
└── ds-0040/                                            a dataset, named after its BoneHub ID
    ├── dataset.json
    └── sub-000001/                                     a subject: one person
        ├── subject.json
        └── scan-01/                                    a scan: one image of the subject
            ├── scan.json
            ├── image.nii.gz
            ├── segmentation.ds-0040.seg.nrrd           the segmentation each provider dataset gave the scan
            ├── segmentation.ds-0018.seg.nrrd
            ├── mesh.ds-0040/    <LABEL>.stl            per-label files, kept locally
            ├── nurbs.ds-0040/   <LABEL>.iges, <LABEL>.step
            ├── mesh.ds-0040.zip                        the zips of those folders, which are published
            └── nurbs.ds-0040.zip
```

A folder's name is its ID; no file name or metadata file repeats it. A scan is identified by its path, e.g.
`ds-0040/sub-000001/scan-01`. A subject holds at most 99 scans.

### One person, one subject: providers

Public datasets overlap: the same CT is in several releases, and one dataset's masks are drawn on another's images.
A database stores each person once. Where several datasets hold the same person or image, their data is **merged**:
the subject lives in one dataset folder, its **home** (usually the dataset whose image is used), and every dataset
that gave the scan data is one of its **providers**, named by its provider key `ds-NNNN`. Each file that holds labels
is named after its provider, the home included, so two datasets' segmentations of the same bone are both kept. The
image comes from one provider. A dataset whose every subject lives elsewhere keeps only its `dataset.json`, which its
files' licenses and citation are read from.

### Dataset IDs

A dataset's ID is its **BoneHub ID** in the
[BoneHub registry](https://github.com/BoneHub/Public-Datasets/blob/main/data/bonehub_public_datasets.json) (browse it
in the [Dataset Browser](https://bonehub.github.io/Public-Datasets/)), written with four digits: TotalSegmentator CT,
BoneHub ID 27, is `ds-0027` in every database that holds it. The registry is the only source of dataset IDs:
`validate`, `prepare` and `upload` check every dataset folder and every `derived_from` against it
(`bonehub_format.load_registry` reads it).

### Metadata files

The metadata files hold only what cannot be read from the data files. Image size and spacing, label counts and which
structures touch the image border are computed into the index.

**dataset.json**: `name`, `description`, `url`, `paper`, `citation` (BibTeX), `country`, `release_date`, `version` (of
the source), `license` (per kind of data), `subsets`, `derived_from`, `annotated_labels`, `remarks`, `schema_version`.

**subject.json**: `sources` (the subject's ID in each provider dataset, the home among them), `gender` (`M`, `F`, `O`;
absent when unknown), `remarks`.

**scan.json**: `image` (the provider the image came from), `modality` (a DICOM code: `CT`, `MR`, ...), `age` (years),
`weight` (kg), `height` (cm), `bmi`, `postmortem`, `days_from_baseline`, `registered_to`, `providers`, `remarks`.
Everything that can change between scans of a person is here, not in subject.json. Each provider gives its
`source_scan_id`, and where they apply its `subset`, `source_split`, `license`, `segmentation`, `mesh`, `nurbs` and
`remarks`:

```json
{
    "image": "ds-0040",
    "modality": "CT",
    "age": 61,
    "providers": {
        "ds-0018": {"source_scan_id": "AutoPET_01140d52d8_56839.nii.gz", "segmentation": {"SKULL": "source", "SACRUM": "source"}},
        "ds-0040": {"source_scan_id": "PETCT_01140d52d8/08-13-2005-NA-PET-CT Ganzkoerper  primaer mit KM-56839"},
        "ds-0134": {"source_scan_id": "0001", "segmentation": {"SACRUM": "source", "GLUTEUS_MAXIMUS_LEFT": "source"}}
    }
}
```

- **Labels and origins.** A provider's `segmentation`, `mesh` and `nurbs` map each label it gave data of that kind for
  to its origin: `source` (it came with the provider's source dataset), `manual` (drawn by hand),
  `model:<name>@<version>`, or `derived:<tool>@<version>` (computed from another file of the same scan, e.g. a mesh
  from the segmentation). A label that is not listed has no data of that kind from that provider.
- **Coverage.** `annotated_labels` lists the labels a dataset's segmentations cover. In a scan with a segmentation, a
  covered label that is missing is absent from the image; a label that is not covered was not segmented. An empty
  list means nothing is segmented; no list means it is unknown or differs between scans.
- **Licenses.** A license is given per kind of data, `{"image": ..., "segmentation": ..., "mesh": ..., "nurbs": ...}`,
  each an SPDX identifier (`CC-BY-4.0`, `CC-BY-NC-SA-4.0`, `MIT`, ... see `bonehub_format.licenses.LICENSES`) or
  `restricted`, for data under a custom license with restricted access. A file is under the license of its kind and
  provider: the provider's entry in scan.json where it gives one, else the provider's subset, else the provider's
  dataset.json. Restricted data is converted like any other but is not published, file by file: a scan can publish
  the open masks drawn on a restricted image.
- **Subsets.** A dataset that collects data from several sources with terms of their own (e.g. CADS) lists them in
  `subsets`, each with its name, URL, citation and licenses; a scan's provider entry names its subset.
- **Derived datasets.** `derived_from` lists the BoneHub IDs of the datasets a dataset was made from, wholly or in
  part. Where a scan was made from a scan of a BoneHub-format dataset, its `source_scan_id` is that scan's path, e.g.
  `ds-0027/sub-000001/scan-01`.
- **Registration.** `registered_to` names the scan of the same subject an image was registered to.

### Data files

- **Images** are NIfTI, on the grid of the source image (never resampled by the conversion). CT is stored as int16 when
  that is lossless, which it is for HU values; float64 images are stored as float32.
- **Segmentations** are 3D Slicer segmentation files (`.seg.nrrd`), one per provider, on the grid of the image; see
  below.
- **Meshes** (`.stl`) and **NURBS surfaces** (`.iges`, `.step`) are in the physical space of the image, in LPS and
  millimetres, one file per label and file type. A scan keeps each provider's in `mesh.ds-NNNN/` and
  `nurbs.ds-NNNN/`; the zips of these folders are what gets published. A scan without an image (e.g. of a dataset that
  publishes only meshes) has meshes in the coordinates its source gives.

### Rules

- **IDs are reproducible.** Subjects are numbered in the natural order of their ID in the home dataset (`case_2`
  before `case_10`), and the scans of a subject by `days_from_baseline` and then `source_scan_id`. Converting the same
  sources again gives the same IDs.
- **Merging is explicit.** Which subjects and images of two datasets are the same is decided by the converters, from
  matching tables kept with them (by source ID where the datasets keep one, else by comparing the images), never
  guessed at conversion time.
- **Changing a dataset is your choice.** A change such as segmenting another bone, correcting a mask or adding meshes
  can be made **in place**: the dataset keeps its ID, and the origin of each label in scan.json tells the new labels
  from the source's (see [Changing a dataset](#changing-a-dataset)). Or it can be published as a **new dataset**: add
  an entry for it to the registry, convert or copy it under its new BoneHub ID, and name the original in `derived_from`.
  Fixing a converter means converting the dataset again under the same ID; this replaces the dataset folder, so
  changes made in place since then must be made again.
- **Versions.** `schema_version` in dataset.json is the version of this format (major.minor). A minor version adds
  something datasets of older minor versions still satisfy, such as a new optional field or a new kind of per-label
  data, which gets a new file or zip in the scan folder and a new label-to-origin block in scan.json. A major version
  makes older datasets invalid. A package reads datasets of its own major version and an equal or lower minor version.

### The index

`index/` holds three Parquet tables, rebuilt from everything above by `python -m bonehub_format index DATABASE`:

- `datasets.parquet`: one row per dataset, with its metadata, its licenses per kind, the number of subjects and
  scans in its folder and the number of scans it gives data to (`scans_provided`);
- `scans.parquet`: one row per scan, with the fields of its subject, its providers, the image's provider, license and
  whether it is published, the image size and spacing, its labels per kind (of all providers), and the segmented
  labels that touch the image border (`truncated`);
- `labels.parquet`: one row per scan, provider, label and kind, with the origin, the license and whether it is
  published.

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
    print(scan.key, scan.info.modality, scan.subject.info.gender, scan.image_path, scan.license("image"))
    for dataset_id, path in scan.segmentation_paths().items():  # one per provider dataset
        print(dataset_id, scan.info.labels("segmentation", dataset_id), scan.license("segmentation", dataset_id), path)
    for dataset_id in scan.provider_ids:
        for label, files in scan.model_files("mesh", dataset_id).items():  # from mesh.ds-NNNN/, or its zip
            stl_bytes = files[0].read_bytes()
```

## Changing a dataset

To change a dataset in place, `update_segmentation` and `update_model_files` add, replace and remove the labels of a
scan and keep its scan.json in step with the files. They change the files of one provider: the scan's home dataset,
or the one given as `dataset_id`.

```python
from bonehub_format import Database, update_model_files, update_segmentation

scan = Database("path/to/database").dataset(27).subject(1).scan(1)

# Add or replace the labels a model segmented. The mask holds LabelMap values on the grid of the scan's image;
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

from bonehub_format import LabelMap as LM, DatasetInfo, Licenses, ScanInfo, SubjectInfo
from bonehub_format.converter import DatasetConverter, ScanSource, SubjectSource
from bonehub_format.export import export_nii_segmentation

LABELS = {1: LM.FEMUR_LEFT.value, 2: LM.FEMUR_RIGHT.value}


def read_dataset(root: Path) -> list[SubjectSource]:
    return [
        SubjectSource(
            source_subject_id=case.name,
            info=SubjectInfo(gender="F"),
            scans=[
                ScanSource(
                    source_scan_id=case.name,
                    info=ScanInfo(modality="CT", age=63),
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
        info = DatasetInfo(
            name="My dataset", license=Licenses.all("CC-BY-4.0"), annotated_labels=["FEMUR_LEFT", "FEMUR_RIGHT"]
        )
        super().__init__(dataset_root, info)
        self.custom_data_handlers.read_dataset = read_dataset
        self.custom_data_handlers.export_segmentation = export_segmentation


if __name__ == "__main__":  # subjects are converted in worker processes, which re-import this script
    MyDataset(Path("path/to/source")).export_to_bonehub_format(Path("path/to/database"), output_dataset_id=27)
```

To merge the data another dataset has of the same image, the home converter adds a `Contribution` to the scan: the
provider's BoneHub ID, its `source_scan_id`, its files and the provider converter's own handlers, which write them
into the files named after the provider. The subject's IDs in the providers go in `SubjectSource.other_sources`, and
the provider's own converter leaves those subjects out (a dataset left with no subjects writes only its
dataset.json):

```python
from bonehub_format.converter import Contribution, CustomDataHandlers
from atlas import export_segmentation as export_atlas_segmentation  # the provider converter's handler

scan.contributions.append(
    Contribution(
        dataset_id=18,
        source_scan_id="AutoPET_01140d52d8_56839.nii.gz",
        segmentation_paths=[atlas_mask],
        handlers=CustomDataHandlers(export_segmentation=export_atlas_segmentation),
    )
)
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
- `LICENSE.md`, the licenses of every dataset per kind of data, and of the subsets and scans with their own;
- `.gitattributes` and `.gitignore`;
- `upload.md`, the steps to upload (it is not uploaded itself);
- `github-webpage/`, the database's webpage, with the steps to publish it on GitHub Pages in its README. Pass
  `--webpage-options options.json` to set its subtitle, About and Contact sections, and analytics snippet (see
  `bonehub_format.webpage.build_webpage`).

`upload` sends the database to HuggingFace without `upload.md`, the logs, the per-label model folders (their zips
are uploaded) and every file whose license is restricted or not known. Use it rather than `hf upload`, which does not read
`.gitignore`. Once the repository is public, add it to the [Harmonized Databases](https://huggingface.co/collections/BoneHub/harmonized-databases) collection.

Other commands: `validate DATABASE [--publishing]`, `pack DATABASE`, `unpack DATABASE` (extract the zips of a
downloaded copy into folders), `index DATABASE`. `validate`, `prepare` and `upload` read the registry from GitHub;
pass `--registry path/to/registry.json` to use a local copy, or `validate --registry none` to skip the ID check.

## Segmentation files

Masks are stored as `.seg.nrrd`, 3D Slicer's segmentation format. Voxels hold per-file segment numbers (1, 2, 3 ...),
and the header maps each number to its `LabelMap` label and records its colour and bounding box. Open them in 3D
Slicer as a **Segmentation**; in Python, `read_segmentation` returns the mask as `LabelMap` values:

```python
import SimpleITK as sitk, numpy as np
from bonehub_format import LabelMap, read_segmentation  # needs the [io] extra

mask = read_segmentation("segmentation.ds-0027.seg.nrrd")  # SimpleITK image of LabelMap values
arr = sitk.GetArrayViewFromImage(mask)
print([LabelMap(v).name for v in np.unique(arr) if v])
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

The structure is `R S M M`: a body region R (1 head, 2 spine and back, 3 thorax and trunk wall, 4 shoulder and arm,
5 hand, 6 pelvis and hip, 7 leg, 8 foot, 9 whole body), a family S within it and a member MM. Families 1–6 are bones,
7 cartilage (articular cartilage, intervertebral discs, costal cartilages), 8 ligaments and 9 muscles, so the map also
holds the soft tissue around the bones: e.g. `RECTUS_FEMORIS_LEFT`, `INTERVERTEBRAL_DISC_L4_L5`,
`ANTERIOR_CRUCIATE_LIGAMENT_RIGHT`. Region 9 holds the whole-body aggregates (`SKELETON`) and the body-composition
classes (`SKELETAL_MUSCLE`, `SUBCUTANEOUS_ADIPOSE_TISSUE`, `SKIN`, ...). `kind_of(v)` says which kind a label is:
`bone`, `cartilage`, `ligament`, `muscle` or `tissue`. The docstring of `LabelMap` has the full rules.

## Related repositories

| Repository | Purpose |
| --- | --- |
| [Quality-Check-Server](https://github.com/BoneHub/Quality-Check-Server) | Dockerised server for the quality check of a database: it hands out scans to reviewers in the browser and to editors in 3D Slicer, and writes approved corrections into the database |
| [Quality-Check-3DSlicer-Extension](https://github.com/BoneHub/Quality-Check-3DSlicer-Extension) | 3D Slicer extension editors use to fetch, correct and submit segmentations |

## License
MIT
