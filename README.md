# BoneHub-Format

The BoneHub format is a folder structure for harmonized databases of 3D medical data, bones first: medical images,
segmentations (of bones and of the rest of the body: organs, vessels, muscles, lesions, devices), and the meshes and
NURBS surfaces made from them, gathered from many source datasets under one label map. This package defines the format and provides the tools to build, check and publish databases in it:

- **schema**: the metadata files, the label map and the segmentation file format;
- **conversion**: a base class for converting a source dataset, and helpers for images, NIfTI and DICOM segmentations;
- **checking**: a validator, which also checks the sources of every dataset against the BoneHub registry, and an
  index of every scan, file and label;
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
and its own index and webpage. A **dataset** is a folder of the database, numbered within it, made from one or more
**sources**: datasets of the [BoneHub registry](https://bonehub.github.io/Public-Datasets/).

```
<database>/
├── README.md, LICENSE.md, .gitattributes, .gitignore   the HuggingFace files (generated)
├── upload.md                                           how to upload the database (generated, not uploaded)
├── index/                                              datasets / sources / scans / files / labels .parquet (generated)
├── github-webpage/                                     the database's webpage (generated, for GitHub Pages; not uploaded)
├── logs/                                               conversion logs (not uploaded)
└── ds-0001/                                            a dataset, numbered within the database
    ├── dataset.json                                    the dataset and its sources (registry datasets)
    └── sub-000001/                                     a subject: one person
        ├── subject.json
        └── scan-01/                                    a scan: one image of the subject
            ├── scan.json
            ├── image.nii.gz         + image.json       the image, and where it came from
            ├── segmentation-01.nrrd  + segmentation-01.json
            ├── segmentation-02.nrrd  + segmentation-02.json   another source's segmentation of the same image
            ├── mesh-01/   <LABEL>.stl                   per-label files, kept locally
            ├── mesh-01.zip           + mesh-01.json      the zip of that folder, which is published
            └── nurbs-01/  <LABEL>.iges, <LABEL>.step -> nurbs-01.zip + nurbs-01.json
```

A folder's name is its ID. A scan is identified by its path, e.g. `ds-0001/sub-000001/scan-01`. A subject holds at
most 99 scans. Every data file has a JSON file of the same name that says which source it came from, its ID there,
its license where it differs from its source's, and (for segmentations, meshes and NURBS) its labels. The files of
one kind are numbered within the scan, whatever their source; no file name names a source.

### One person, one subject

Public datasets overlap: the same CT is in several releases, and one dataset's masks are drawn on another's images.
A database stores each person once. Where several sources hold the same person or image, their data is **merged**
into one subject of one dataset: e.g. ENHANCE.PET's CT of an autoPET study, with ENHANCE's segmentation
(`segmentation-01`) and Atlas's segmentation of the same CT (`segmentation-02`). The dataset lists every source it draws
from in its `dataset.json`; a source whose data all went into another dataset's scans has no folder of its own.

A database aims at many different people, not at many images of the same person: of the contrast phases,
reconstructions, follow-ups or test/retest scans a source has of one person, **one scan is converted**, the one with
the most labelled data, then the one covering most of the body, with the thinnest slices, the earliest
(`bonehub_format.selection`). A subject has a second scan only where it holds other data that cannot be put on
the first scan's grid, such as the cryosection images of the Visible Human next to its CT.

### Dataset numbers and sources

A dataset's number (`ds-0001`) means nothing outside its database; whoever builds the database chooses it. What a
dataset is made from is in its `dataset.json`: its **sources**, keyed by their **BoneHub ID** in the
[registry](https://github.com/BoneHub/Public-Datasets/blob/main/data/bonehub_public_datasets.json) (browse it in the
[Dataset Browser](https://bonehub.github.io/Public-Datasets/)). TotalSegmentator CT is source 27 in every database
that holds it. `validate`, `prepare` and `upload` check every source against the registry
(`bonehub_format.load_registry` reads it).

### Metadata files

The metadata files hold only what cannot be read from the data files. Image size and spacing, label counts, file
sizes and which structures touch the image border are computed into the index.

**dataset.json**: `name`, `description` (how it was made from its sources), `sources`, `remarks`, `schema_version`.
Each source (by BoneHub ID) has `name`, `description`, `url`, `paper`, `country`, `release_date`, `version`,
`license` (per kind of data), `subsets`, `annotated_labels`, `remarks`.

**subject.json**: `sources` (the subject's ID in each source, by BoneHub ID), `gender` (`M`, `F`, `O`; absent when
unknown), `race` and `ethnicity` (in the words of the source, where it records them), `remarks`.

**scan.json**: `modality` (a DICOM code: `CT`, `MR`, `XC`, ...), `age` (years), `weight` (kg), `height` (cm), `bmi`,
`postmortem`, `days_from_baseline`, `registered_to`, `remarks`. Everything that can change between scans of a person
is here, not in subject.json.

**image.json**, **segmentation-NN.json**, **mesh-NN.json**, **nurbs-NN.json**: `source` (its BoneHub ID),
`source_scan_id`, and where they apply `subset`, `source_split`, `license`, `remarks`, and `labels` (label -> origin):

```json
{
    "source": 18,
    "source_scan_id": "AutoPET_01140d52d8_56839.nii.gz",
    "labels": {"SKULL": "source", "SACRUM": "source", "FEMUR_LEFT": "source"}
}
```

- **Labels and origins.** `labels` maps each label of the file to its origin: `source` (it came with the source
  dataset), `manual` (drawn by hand), `model:<name>@<version>`, or `derived:<tool>@<version>` (computed from another
  file of the same scan, e.g. a mesh from the segmentation).
- **Coverage.** `annotated_labels` of a source lists the labels its segmentations cover. In a scan with a segmentation
  of that source, a covered label that is missing is absent from the image; a label that is not covered was not
  segmented. An empty list means nothing is segmented; no list means it is unknown or differs between scans.
- **Licenses.** A license is given per kind of data, `{"image": ..., "segmentation": ..., "mesh": ..., "nurbs": ...}`,
  each an SPDX identifier (`CC-BY-4.0`, `CC-BY-NC-SA-4.0`, `MIT`, ... see `bonehub_format.licenses.LICENSES`) or
  `restricted`, for data under a custom license with restricted access. A file is under its own license where its JSON
  gives one, else its source's subset's, else its source's, for its kind. Restricted data is converted like any other
  but is not published, file by file: a scan can publish the open masks drawn on a restricted image.
- **Subsets.** A source that collects data from several collections with terms of their own (e.g. CADS) lists them in
  `subsets`, each with its name, URL and licenses; a file names its subset.
- **Registration.** `registered_to` names the scan of the same subject an image was registered to.

### Data files

- **Images** are NIfTI, on the grid of the source image (never resampled by the conversion). CT is stored as int16 when
  that is lossless, which it is for HU values; float64 images are stored as float32.
- **Segmentations** are NRRD files on the grid of the image, whose voxels hold segment numbers and whose header maps
  them to labels (see [Segmentation files](#segmentation-files)). A mask that holds the image's voxels in another order
  (an axis stored flipped) or with an origin off by less than half a voxel (header rounding) is written again on the
  image grid at conversion, which moves no label; a mask on any other grid fails the subject.
- **Meshes** (`.stl`) and **NURBS surfaces** (`.iges`, `.step`) are in the physical space of the image, in LPS and
  millimetres, one file per label and file type. A scan keeps them in `mesh-NN/` and `nurbs-NN/`; the zips of these
  folders are what gets published. A scan without an image (e.g. of a source that publishes only meshes) has meshes in
  the coordinates its source gives.

### Rules

- **IDs are reproducible.** Subjects are numbered in the natural order of their ID in the converted source (`case_2`
  before `case_10`), and the scans of a subject by `days_from_baseline`, then the source's own scans before those only
  other sources give, then `source_scan_id`. Converting the same sources again gives the same IDs.
- **Merging is explicit.** Which subjects and images of two sources are the same is decided by the converters, from
  matching tables kept with them (by source ID where the sources keep one, else by comparing the images), never
  guessed at conversion time.
- **Changing a dataset is your choice.** A change such as segmenting another bone, correcting a mask or adding meshes
  can be made **in place**: the origin of each label tells the new labels from the source's (see
  [Changing a dataset](#changing-a-dataset)). Or it can be published as a **new source**: add an entry for it to the
  registry and convert it. Fixing a converter means converting the dataset again; this replaces the dataset folder, so
  changes made in place since then must be made again.
- **Versions.** `schema_version` in dataset.json is the version of this format (major.minor). A minor version adds
  something datasets of older minor versions still satisfy, such as a new optional field or a new kind of per-label
  data. A major version makes older datasets invalid. A package reads datasets of its own major version and an equal
  or lower minor version.

### The index

`index/` holds five Parquet tables, rebuilt from everything above by `python -m bonehub_format index DATABASE`:

- `datasets.parquet`: one row per dataset, with its name, sources and numbers of subjects and scans;
- `sources.parquet`: one row per dataset and source, with the source's metadata, licenses per kind, and the numbers
  of files and scans it gave;
- `scans.parquet`: one row per scan, with the fields of its subject, its sources, the image's source, license and
  whether it is published, the image size and spacing, its labels per kind (of all files), and the segmented labels
  that touch the image border (`truncated`);
- `files.parquet`: one row per data file, with its path, size, source, license and whether it is published (its
  direct download link on HuggingFace is `https://huggingface.co/datasets/<repo>/resolve/main/<path>`);
- `labels.parquet`: one row per data file and label, with the origin, the license and whether it is published.

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
    print(scan.key, scan.info.modality, scan.subject.info.gender, scan.image_path, scan.sources)
    for segmentation in scan.files("segmentation"):  # segmentation-01.nrrd, segmentation-02.nrrd, ...
        print(segmentation.path, segmentation.source, segmentation.labels, segmentation.license())
    for mesh in scan.files("mesh"):
        for label, files in mesh.model_files().items():  # from mesh-NN/, or its zip
            stl_bytes = files[0].read_bytes()
```

## Changing a dataset

To change a dataset in place, `update_segmentation` and `update_model_files` add, replace and remove the labels of a
scan and keep the JSON files in step with the data files. They change one numbered file (`number`), or write a new
one when none is given, whose JSON names `source` (the dataset's first source by default).

```python
from bonehub_format import Database, update_model_files, update_segmentation

scan = Database("path/to/database").dataset(1).subject(1).scan(1)

# Add or replace the labels a model segmented, in segmentation-01: label -> its mask on the grid of the scan's image
# (or the path of a BoneHub segmentation). Where it overlaps another label, both are kept, in separate layers;
# overlap="replace" or "keep" gives the voxel to the new label or to the one already there.
update_segmentation(scan, {"PATELLA_LEFT": patella_mask}, origin="model:MyModel@1.0", number=1)
update_segmentation(scan, remove=["PATELLA_LEFT"], number=1)

update_model_files(scan, "mesh", {"PATELLA_LEFT": Path("patella.stl")}, origin="derived:my-mesher@1.0", number=1)
```

A file left without labels is deleted, and the files after it are numbered again. Neither changes dataset.json: when
a label is now segmented in every scan of a source, add it to its `annotated_labels`. Rebuild the index afterwards.
`update_segmentation` needs the `io` extra.

## Converting a dataset

A converter subclasses `DatasetConverter`, names its source (BoneHub ID and `SourceInfo`) and sets its handlers.
`read_dataset` lists the subjects and their scans; the image is read from `image_path` (a file or a DICOM folder)
unless `export_image` is set:

```python
from pathlib import Path

from bonehub_format import LabelMap as LM, Licenses, ScanInfo, SourceInfo, SubjectInfo
from bonehub_format.converter import DatasetConverter, ScanSource, SubjectSource
from bonehub_format.export import export_nii_segmentation

# source label -> its label; a list where the source label is part of several labels (KiTS's tumor is part of the
# kidney: [LM.KIDNEY, LM.KIDNEY_TUMOR]), Instance(label, k) for one of several segments of a label (a lesion)
LABELS = {1: LM.FEMUR_LEFT, 2: LM.FEMUR_RIGHT}


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
        source = SourceInfo(name="My dataset", license=Licenses.all("CC-BY-4.0"), annotated_labels=["FEMUR_LEFT", "FEMUR_RIGHT"])
        super().__init__(dataset_root, 27, source)  # 27: the source's BoneHub ID
        self.custom_data_handlers.read_dataset = read_dataset
        self.custom_data_handlers.export_segmentation = export_segmentation


if __name__ == "__main__":  # subjects are converted in worker processes, which re-import this script
    MyDataset(Path("path/to/source")).export_to_bonehub_format(Path("path/to/database"), output_dataset_id=1)
```

To merge the data another source has of the same image, the converter adds a `Contribution` to the scan: the source's
BoneHub ID, its `source_scan_id`, its files and the export handlers of that source, which write them as the scan's
next numbered files. It passes that source's `SourceInfo` in `other_sources`, so that the dataset lists it, and puts
the subject's ID there in `SubjectSource.other_sources`:

```python
from bonehub_format.converter import Contribution, CustomDataHandlers

scan.contributions.append(
    Contribution(
        source_id=18,
        source_scan_id="AutoPET_01140d52d8_56839.nii.gz",
        segmentation_paths=[atlas_mask],
        handlers=CustomDataHandlers(export_segmentation=export_atlas_segmentation),
    )
)
```

A subject's folder is written completely or not at all, and `subject.json` is written last. `overwrite=True`
converts a dataset again from scratch; `resume=True` keeps the subjects an earlier run finished and converts the rest,
after checking that the source still lists the same subjects. A source that lists no subjects writes nothing. Logs go
to `<database>/logs/`.

The converters of the BoneHub source datasets are in the private `custom_dataset_io` submodule.

## Publishing

```bash
python -m bonehub_format prepare DATABASE --repo-id BoneHub/<database> --title "<title>" --homepage https://bonehub.github.io/<database>
python -m bonehub_format upload DATABASE --repo-id BoneHub/<database>
```

`prepare` brings the zips up to date, rebuilds the index, validates the database, and writes:

- `README.md`, the dataset card, which the HuggingFace dataset viewer shows with the index tables; it explains how
  to download the data (all of it, a dataset, or chosen scans) and how to read it;
- `LICENSE.md`, the licenses of every source per kind of data, and of the collections and files with their own;
- `.gitattributes` and `.gitignore`;
- `upload.md`, the steps to upload (it is not uploaded itself);
- `github-webpage/`, the database's webpage: an overview of the database and a table of every scan with filters and
  a direct download link per file, with the steps to publish it on GitHub Pages in its README. Pass
  `--webpage-options options.json` to set its subtitle, About and Contact sections, and analytics snippet (see
  `bonehub_format.webpage.build_webpage`).

`upload` sends the database to HuggingFace without `upload.md`, the logs, the webpage (it goes to GitHub Pages), the
per-label model folders (their zips are uploaded) and every file whose license is restricted or not known. Use it rather than `hf upload`, which does not read
`.gitignore`. Once the repository is public, add it to the [Harmonized Databases](https://huggingface.co/collections/BoneHub/harmonized-databases) collection.

Other commands: `validate DATABASE [--publishing]`, `pack DATABASE`, `unpack DATABASE` (extract the zips of a
downloaded copy into folders), `index DATABASE`. `validate`, `prepare` and `upload` read the registry from GitHub;
pass `--registry path/to/registry.json` to use a local copy, or `validate --registry none` to skip the ID check.

## Segmentation files

Masks are stored as NRRD files (`segmentation-NN.nrrd`), in the fields 3D Slicer uses for its segmentations (load the
file as a *Segmentation* in Slicer's Add Data dialog). Voxels hold per-file segment numbers (1, 2, 3 ...), and the
header gives each segment its label name (`Segment<i>_Name`, and the tag `BoneHubLabel`), its colour and its bounding
box. Labels that overlap — a kidney and its cyst, two models' masks of one region — are kept whole in separate
layers: the file is then 4D, its first axis the layers (`kinds: list domain domain domain`), as Slicer writes
overlapping segments. Several segments may share a label, told apart by their instance (`BoneHubInstance`: lesion 2,
the second fragment of a broken bone) or rater (`BoneHubRater`).

```python
import SimpleITK as sitk
from bonehub_format import read_label_mask, read_segmentation  # needs the [io] extra

femur = read_label_mask("segmentation-01.nrrd", "FEMUR_LEFT")  # 0/1 uint8 image: the union of the label's segments
layers, segments = read_segmentation("segmentation-01.nrrd")   # one image of segment numbers per layer, and
for s in segments:                                             # each segment's label, layer, number, instance, rater
    print(s.id, s.layer, s.number)                             # FEMUR_LEFT 0 1, KIDNEY_CYST_LEFT 1 1, LIVER_TUMOR#2 0 3
```

`write_segmentation({label: mask, ...}, reference_image, path)` writes one, putting overlapping masks into layers.
`read_segmentation_labels` lists the labels of a mask from its header alone, without numpy or SimpleITK.

## Label map

A label is a name; `LabelMap` lists them all (2,400 and more), and `LabelMap.FEMUR_LEFT.value == "FEMUR_LEFT"`. A
name says which structure, part, tissue and side it is, always in this order:

```
NODE[_PART][_TISSUE][_SIDE]

FEMUR_LEFT                      the left femur
FEMUR_PROXIMAL_CORTICAL_LEFT    the cortical bone of the proximal femur, left
VERTEBRA_L3_BODY                the body of the third lumbar vertebra
KIDNEY_CYST_LEFT                a cyst of the left kidney (findings and devices are tissues of their structure)
LEFT_ATRIUM_LUMEN               the blood pool of the left atrium
```

The structures form a tree with one root per kind: SKELETON, CARTILAGE, LIGAMENTS, JOINTS, MUSCLES,
HEART_AND_PERICARDIUM, ARTERIES, VEINS, LYMPHOID_ORGANS, BRAIN, SPINAL_CORD, CEREBROSPINAL_FLUID, NERVES,
EYE_AND_ADNEXA, EAR, RESPIRATORY_SYSTEM, DIGESTIVE_SYSTEM, URINARY_SYSTEM, GENITAL_ORGANS, ENDOCRINE_GLANDS,
INTEGUMENT, ADIPOSE_TISSUE, BODY_CAVITIES and BODY. A name without a side is both sides together (or a structure that
is not paired), and a label always means the whole of what its name says: the kidney with its cysts and tumors.
Aggregates are unions with a name of their own (`RIBS`, `SKULL_WITHOUT_MANDIBLE`, `JEJUNUM_AND_ILEUM`).

`node_of`, `part_of`, `tissue_of` and `side_of` take a name apart, `kind_of` and `category_of` say what it is
(`bone`, `muscle`, `heart` ...; `tissue`, `finding` or `device`), and `covers(a, b)` says whether label `a` holds
label `b` — `covers("RIBS", "RIB_7_LEFT")`, `covers("KIDNEY_LEFT", "KIDNEY_CYST_LEFT")`,
`covers("FEMUR_PROXIMAL", "FEMUR_HEAD_LEFT")` — so a search for a structure finds the scans of everything in it;
`labels_covered(a)` lists them. The tree is written by hand in `bonehub_format/labelmap_spec.py`;
`tools/generate_labelmap.py` writes the enum (`labelmap_data.py`) and checks every rule. The label map only grows: a
published label keeps its name and meaning, and what it covers among published labels never changes
(`tools/generate_labelmap.py --publish` records them; the check then holds the map to them).

## Related repositories

| Repository | Purpose |
| --- | --- |
| [Quality-Check-Server](https://github.com/BoneHub/Quality-Check-Server) | Dockerised server for the quality check of a database: it hands out scans to reviewers in the browser and to editors in 3D Slicer, and writes approved corrections into the database |
| [Quality-Check-3DSlicer-Extension](https://github.com/BoneHub/Quality-Check-3DSlicer-Extension) | 3D Slicer extension editors use to fetch, correct and submit segmentations |

## License
MIT
