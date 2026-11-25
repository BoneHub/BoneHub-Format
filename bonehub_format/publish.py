"""The files that make a database a Hugging Face dataset repository.

    README.md        the dataset card: what the data is, how it is laid out and how to download parts of it
    LICENSE.md       the licenses of each dataset per kind of data, and of the subsets and scans that have their own
    .gitattributes   which files Hugging Face stores as large files
    .gitignore       what stays out of the upload: upload.md, logs, the per-label model folders (their zips are
                     uploaded) and every file whose license is restricted or not known
    upload.md        how to upload the database; it is not uploaded itself

Whether a file is published is decided per file, from the license of its kind of data and its provider (see
reader.Scan.license): a scan can publish its segmentation and keep its restricted image back.

write_hub_files() writes them all from the metadata, after build_index() has written the index the card points to.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import layout
from .index import INDEX_FILES
from .licenses import LICENSE_KINDS, RESTRICTED, is_publishable, license_name, license_url
from .metadata import LABEL_KINDS
from .reader import Database
from .registry import REGISTRY_PAGE_URL, registry_entry_url

FORMAT_URL = "https://github.com/BoneHub/BoneHub-Format"
LABEL_MAP_URL = f"{FORMAT_URL}/blob/main/bonehub_format/labelmap.py"

_GITATTRIBUTES = """\
# Files stored as large files on Hugging Face. Written by bonehub_format.publish.
*.nii filter=lfs diff=lfs merge=lfs -text
*.nii.gz filter=lfs diff=lfs merge=lfs -text
*.nrrd filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.stl filter=lfs diff=lfs merge=lfs -text
*.iges filter=lfs diff=lfs merge=lfs -text
*.step filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
github-webpage/data.js filter=lfs diff=lfs merge=lfs -text
"""

_MODALITY_TAGS = {"CT": "ct", "MR": "mri", "MRI": "mri", "PT": "pet", "CBCT": "cbct", "US": "ultrasound"}


def _size_category(count: int) -> str:
    for limit, name in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"), (1_000_000, "100K<n<1M")):
        if count < limit:
            return name
    return "1M<n<10M"


def _yaml(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)  # a JSON string is a valid YAML scalar


def _license_link(value: str | None) -> str:
    if value is None:
        return "not given"
    url = license_url(value)
    return f"[{value}]({url})" if url else license_name(value)


def scan_files(scan) -> list[tuple[str, str, int, str | None]]:
    """(file name, kind, provider dataset ID, license) of every data file of a scan that would be published: the
    image, and each provider's segmentation and model zips."""
    files = []
    if scan.image_path is not None:
        files.append((layout.IMAGE_FILE, "image", scan.info.image_provider, scan.license("image")))
    for dataset_id in scan.provider_ids:
        provider = scan.info.provider(dataset_id)
        if provider.labels("segmentation"):
            path = layout.segmentation_file(scan.path, dataset_id)
            files.append((path.name, "segmentation", dataset_id, scan.license("segmentation", dataset_id)))
        for kind in layout.MODEL_KINDS:
            if provider.labels(kind):
                path = layout.model_zip(scan.path, kind, dataset_id)
                files.append((path.name, kind, dataset_id, scan.license(kind, dataset_id)))
    return files


def _withheld(root: Database) -> list[str]:
    """The data files that are not published (restricted or unknown license), as paths relative to the root; a file
    name withheld in every scan of a dataset is given as one pattern, e.g. ds-0040/*/*/image.nii.gz."""
    by_name: dict[tuple[str, str], list[tuple[str, bool]]] = {}  # (dataset folder, file name) -> (scan key, published)
    for scan in root.scans():
        for name, _, _, license in scan_files(scan):
            by_name.setdefault((scan.dataset.path.name, name), []).append((scan.key, is_publishable(license)))
    withheld = []
    for (dataset_folder, name), entries in sorted(by_name.items()):
        if not any(published for _, published in entries):
            withheld.append(f"{dataset_folder}/*/*/{name}")
        else:
            withheld += [f"{key}/{name}" for key, published in entries if not published]
    return withheld


def excluded_from_upload(root: Path) -> list[tuple[str, str]]:
    """What stays out of the upload, as (.gitignore pattern, upload pattern) pairs.

    Upload patterns are matched with fnmatch against paths relative to the database's root folder, as huggingface_hub matches its
    ignore_patterns: '*' also matches across folders.
    """
    reader = root if isinstance(root, Database) else Database(root)
    excluded = [
        ("upload.md", "upload.md"),
        (f"{layout.LOGS_DIR}/", f"{layout.LOGS_DIR}/*"),
        (".bonehub_qc/", ".bonehub_qc/*"),
        ("__pycache__/", "*__pycache__/*"),
        ("*.tmp", "*.tmp"),
        *((f"**/{kind}.ds-*/", f"*/{kind}.ds-*/*") for kind in layout.MODEL_KINDS),
    ]
    excluded += [(f"/{path}", path) for path in _withheld(reader)]
    return excluded


def _gitignore(root: Database) -> str:
    lines = [
        "# What stays out of the Hugging Face upload. Written by bonehub_format.publish, from the licenses in the",
        "# metadata; write it again with write_hub_files rather than editing it. upload() passes the same exclusions",
        "# to Hugging Face explicitly, because its upload does not read this file.",
    ]
    for pattern, _ in excluded_from_upload(root):
        if pattern == "**/mesh.ds-*/":
            lines.append("# Per-label model files: their zips (mesh.ds-NNNN.zip, nurbs.ds-NNNN.zip) are uploaded instead")
        elif pattern.startswith("/ds-") and not any(line.startswith("# Not published") for line in lines):
            lines.append(f"# Not published: files under a restricted license ('{RESTRICTED}') or with no known license")
        lines.append(pattern)
    return "\n".join(lines) + "\n"


def upload(root: Path, repo_id: str, commit_message: str | None = None) -> None:
    """Upload a database to its Hugging Face dataset repository, leaving out what excluded_from_upload lists.

    The repository must exist (hf repos create <repo_id> --repo-type dataset --private) and you must be logged in
    (hf auth login). Running it again resumes an interrupted upload. Requires huggingface_hub.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError('Uploading needs huggingface_hub: pip install -U "huggingface_hub[hf_xet]"') from e
    root = Path(root)
    HfApi().upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=root,
        ignore_patterns=[pattern for _, pattern in excluded_from_upload(root)],
        commit_message=commit_message or "Upload dataset",
    )


class _Summary:
    def __init__(self, root: Database):
        self.datasets = root.datasets()
        self.scans = list(root.scans())
        self.by_dataset = Counter(scan.dataset.id for scan in self.scans)
        self.subjects_by_dataset = Counter(dataset_id for dataset_id, _ in {(s.dataset.id, s.subject.id) for s in self.scans})
        self.provided = Counter(dataset_id for scan in self.scans for dataset_id in scan.provider_ids)
        self.modalities = Counter(scan.info.modality or "unknown" for scan in self.scans)
        self.kind_counts = {kind: sum(len(scan.info.labels(kind)) for scan in self.scans) for kind in LABEL_KINDS}
        # (provider dataset ID, kind) -> (files published, files withheld)
        self.files = Counter()
        self.withheld = Counter()
        self.scan_licenses = Counter()  # (provider, kind, license) of scans and subsets with a license of their own
        for scan in self.scans:
            for _, kind, dataset_id, license in scan_files(scan):
                self.files[(dataset_id, kind)] += 1
                if not is_publishable(license):
                    self.withheld[(dataset_id, kind)] += 1
                provider = scan.info.provider(dataset_id)
                if provider is not None and (provider.license and provider.license.of(kind) or provider.subset):
                    self.scan_licenses[(dataset_id, kind, license)] += 1
        self.restricted = [d for d in self.datasets if any(self.withheld[(d.id, kind)] for kind in LICENSE_KINDS)]


def _licenses_text(info) -> str:
    """A dataset's licenses per kind, in one cell: one license if all kinds share it."""
    if info.license is None:
        return "not given"
    given = {kind: info.license.of(kind) for kind in LICENSE_KINDS if info.license.of(kind) is not None}
    if len(set(given.values())) == 1:
        return _license_link(next(iter(given.values())))
    return "; ".join(f"{kind}: {_license_link(value)}" for kind, value in given.items()) or "not given"


def _card(root: Database, summary: _Summary, repo_id: str, title: str, description: str | None, homepage: str | None) -> str:
    tags = ["medical", "bone", "3d", "bonehub-format"]
    tags += sorted({_MODALITY_TAGS.get(m, m.lower()) for m in summary.modalities if m != "unknown"})
    tags += [kind for kind in layout.MODEL_KINDS if summary.kind_counts[kind]]
    yaml = [
        "---",
        f"pretty_name: {_yaml(title)}",
        "license: other",
        "license_name: per-dataset-licenses",
        "license_link: LICENSE.md",
        "task_categories:",
        "- image-segmentation",
        "tags:",
        *(f"- {tag}" for tag in tags),
        "size_categories:",
        f"- {_size_category(len(summary.scans))}",
        "configs:",
        "- config_name: scans",
        f"  data_files: {layout.INDEX_DIR}/{INDEX_FILES['scans']}",
        "  default: true",
        "- config_name: labels",
        f"  data_files: {layout.INDEX_DIR}/{INDEX_FILES['labels']}",
        "- config_name: datasets",
        f"  data_files: {layout.INDEX_DIR}/{INDEX_FILES['datasets']}",
        "---",
    ]

    subjects = sum(summary.subjects_by_dataset.values())
    modalities = ", ".join(f"{count} {modality}" for modality, count in summary.modalities.most_common())
    kinds = ", ".join(f"{count} {kind} labels" for kind, count in summary.kind_counts.items() if count)
    rows = [
        "| ID | Dataset | Subjects | Scans | Data in scans | License | Source |",
        "|---|---|---|---|---|---|---|",
    ]
    for dataset in summary.datasets:
        info = dataset.info
        source = " ".join(f"[{name}]({url})" for name, url in (("data", info.url), ("paper", info.paper)) if url)
        rows.append(
            f"| [{dataset.id}]({registry_entry_url(dataset.id)}) | {info.name} | {summary.subjects_by_dataset[dataset.id]} "
            f"| {summary.by_dataset[dataset.id]} | {summary.provided[dataset.id]} "
            f"| {_licenses_text(info)} | {source} |"
        )

    body = [
        f"# {title}",
        "",
        description or f"3D bone data in the [BoneHub format]({FORMAT_URL}): medical images, bone segmentations and "
        "the meshes and NURBS surfaces derived from them, converted from public datasets.",
        "",
        f"{len(summary.datasets)} datasets, {subjects} subjects and {len(summary.scans)} scans ({modalities}); {kinds}."
        + (f" Website: {homepage}" if homepage else ""),
        "",
        "## Datasets",
        "",
        *rows,
        "",
    ]
    body += [
        "*Subjects* and *Scans* count the people and scans that live in a dataset's folder; *Data in scans* counts the "
        "scans, in any folder, that hold data the dataset provided (see *Merged datasets* below).",
        "",
    ]
    if summary.restricted:
        lines = []
        for dataset in summary.restricted:
            kinds = [f"{summary.withheld[(dataset.id, kind)]} {kind}" for kind in LICENSE_KINDS if summary.withheld[(dataset.id, kind)]]
            lines.append(f"- {dataset.id} ({dataset.info.name}): {', '.join(kinds)} files")
        body += [
            "Some files are under a restricted license (or none is known), so they are listed in the index but not in "
            "this repository; the `published` and `image_published` columns of the index say which. Obtain them from "
            "their source and convert them with the BoneHub-Format package:",
            "",
            *lines,
            "",
        ]
    body += [
        "## Layout",
        "",
        "```",
        "ds-0040/                                 a source dataset, named after its BoneHub ID",
        "├── dataset.json                         source, licenses, which labels its segmentations cover",
        "└── sub-000001/                          a subject (person)",
        "    ├── subject.json                     the subject's ID in each dataset, gender",
        "    └── scan-01/                         a scan of the subject",
        "        ├── scan.json                    modality, age, weight, height; per provider: ID, labels, origin",
        "        ├── image.nii.gz",
        "        ├── segmentation.ds-0040.seg.nrrd   the segmentation each provider dataset gave",
        "        ├── segmentation.ds-0018.seg.nrrd",
        "        ├── mesh.ds-0040.zip             <LABEL>.stl",
        "        └── nurbs.ds-0040.zip            <LABEL>.iges / <LABEL>.step",
        "index/                                   Parquet tables of all datasets, scans and labels",
        "```",
        "",
        "A folder's name is its ID. The index (also shown in the dataset viewer above) holds every field of the "
        "metadata files, plus image size and voxel spacing, the structures that touch the image border, and the "
        "license of each file.",
        "",
        "## Downloading part of the data",
        "",
        "Choose scans in the index, then download only their files:",
        "",
        "```python",
        "import pandas as pd",
        "from huggingface_hub import snapshot_download",
        "",
        f'scans = pd.read_parquet("hf://datasets/{repo_id}/{layout.INDEX_DIR}/{INDEX_FILES["scans"]}")',
        'femurs = scans[scans["segmentation"].apply(lambda labels: "FEMUR_LEFT" in labels) & scans["image_published"]]',
        "snapshot_download(",
        f'    "{repo_id}", repo_type="dataset", local_dir="data",',
        '    allow_patterns=[f"{key}/*" for key in femurs["scan"]] + ["ds-*/dataset.json", "ds-*/*/subject.json"],',
        ")",
        "```",
        "",
        "or with the command line, e.g. every image and segmentation of dataset 27:",
        "",
        "```",
        f'hf download {repo_id} --repo-type dataset --local-dir data --include "ds-0027/*/image.nii.gz" '
        '--include "ds-0027/*/segmentation.*.seg.nrrd" --include "ds-0027/*.json"',
        "```",
        "",
        "## Conventions",
        "",
        f"- **Labels** are the names of the [BoneHub label map]({LABEL_MAP_URL}). Segmentations are 3D Slicer "
        "segmentation files (`.seg.nrrd`); open them in 3D Slicer, or read them with `bonehub_format.read_segmentation`.",
        "- **Providers**: the data of a scan can come from several datasets. Each file is named after the dataset "
        "that provided it (`segmentation.ds-0018.seg.nrrd`); `scan.json` lists every provider with the scan's ID in "
        "it, the labels it gave and the dataset the image came from. Where two providers segmented the same "
        "structure, both versions are kept.",
        "- **Origin**: every label of a provider says how that dataset came by it: `source` (its source dataset), "
        "`manual`, `model:<name>@<version>` or `derived:<tool>@<version>` (made from another file of the scan, "
        "e.g. a mesh from the segmentation).",
        "- **Coordinates**: meshes and NURBS surfaces are in the physical space of the image, in LPS and millimetres.",
        "- **Coverage**: `annotated_labels` in `dataset.json` lists the labels a dataset's segmentations cover. In a scan "
        "with a segmentation, a covered label that is missing is absent from the image; other labels were not segmented.",
        f"- **IDs**: a dataset's ID is its BoneHub ID in the [BoneHub registry]({REGISTRY_PAGE_URL}), the same in every "
        "BoneHub database; each ID in the table above links to its registry entry.",
        "- **Merged datasets**: each person is stored once. Where several source datasets hold the same person or "
        "image, their data is merged into one subject, which lives in one of them (its home) and lists its ID in "
        "the others in `subject.json`. `derived_from` in `dataset.json` names the datasets a dataset was made from.",
        "",
        "## Licenses",
        "",
        "Each file keeps the license of the dataset that provided it, per kind of data (image, segmentation, mesh, "
        "NURBS): see [LICENSE.md](LICENSE.md), and the `image_license` and `license` columns of the index. Cite the "
        "source datasets you use.",
        "",
        "## Reading the data with Python",
        "",
        "```python",
        "from bonehub_format import Database",
        "",
        'for scan in Database("data").scans():',
        '    print(scan.key, scan.info.modality, scan.image_path, scan.segmentation_paths())',
        "```",
        "",
        f"`pip install bonehub-format @ git+{FORMAT_URL}.git`",
        "",
    ]
    return "\n".join(yaml + body)


def _license_file(summary: _Summary) -> str:
    lines = [
        "# Licenses",
        "",
        "Each file is under the license of the dataset that provided it, for its kind of data: `image.nii.gz` under "
        "the image license of the dataset the image came from, `segmentation.ds-NNNN.seg.nrrd` and the "
        "`mesh.ds-NNNN.zip` / `nurbs.ds-NNNN.zip` files under the segmentation, mesh or NURBS license of dataset NNNN. "
        "Subsets and scans with licenses of their own are listed below the table; `index/scans.parquet` "
        "(`image_license`) and `index/labels.parquet` (`license`) give the license of every file.",
        "",
        "| ID | Dataset | Image | Segmentation | Mesh | NURBS | Source |",
        "|---|---|---|---|---|---|---|",
    ]
    for dataset in summary.datasets:
        info = dataset.info
        cells = [_license_link(info.license.of(kind)) if info.license else "not given" for kind in LICENSE_KINDS]
        lines.append(f"| {dataset.id} | {info.name} | {' | '.join(cells)} | {info.url or ''} |")
    subsets = [
        f"| {dataset.id} | {name} | {subset.name} | "
        + " | ".join(_license_link(subset.license.of(kind)) if subset.license and subset.license.of(kind) else "as the dataset" for kind in LICENSE_KINDS)
        + f" | {subset.url or ''} |"
        for dataset in summary.datasets
        for name, subset in (dataset.info.subsets or {}).items()
    ]
    if subsets:
        lines += [
            "",
            "## Subsets with licenses of their own",
            "",
            "| ID | Subset | Name | Image | Segmentation | Mesh | NURBS | Source |",
            "|---|---|---|---|---|---|---|---|",
            *subsets,
        ]
    overrides = [
        f"- Dataset {dataset_id}, {kind}: {count} files under {_license_link(value)}"
        + ("" if is_publishable(value) else " (not published)")
        for (dataset_id, kind, value), count in sorted(summary.scan_licenses.items(), key=lambda item: (item[0][0], item[0][1], str(item[0][2])))
    ]
    if overrides:
        lines += ["", "## Files under the license of their subset or scan", "", *overrides]
    if summary.restricted:
        lines += [
            "",
            "## Files that are not published",
            "",
            "These files are under a custom license with restricted access, or no license is known for them. They are "
            "listed in the index, but they are not in this repository; obtain them from their source.",
            "",
        ]
        for dataset in summary.restricted:
            kinds = [f"{summary.withheld[(dataset.id, kind)]} {kind}" for kind in LICENSE_KINDS if summary.withheld[(dataset.id, kind)]]
            lines.append(f"- {dataset.id}: {dataset.info.name}: {', '.join(kinds)} files ({dataset.info.url or 'no URL given'})")
    return "\n".join(lines) + "\n"


def _upload_guide(root: Path, summary: _Summary, repo_id: str, title: str) -> str:
    size = 0
    for scan in summary.scans:
        size += (scan.path / layout.SCAN_JSON).stat().st_size
        for name, _, _, license in scan_files(scan):
            if is_publishable(license):
                size += (scan.path / name).stat().st_size
    return f"""# Uploading {title} to Hugging Face

This file is listed in `.gitignore`, so it is not uploaded.

## 1. Prepare the database

```
python -m bonehub_format prepare "{root}" --repo-id {repo_id} --title "{title}"
```

This brings the zips of the model folders (`mesh.ds-NNNN/`, `nurbs.ds-NNNN/`) up to date, rebuilds the index, checks the database, and writes
the card, `LICENSE.md`, `.gitattributes`, `.gitignore`, this file and `github-webpage/`. Fix any problem it reports
before uploading.

## 2. Log in

```
pip install -U "huggingface_hub[hf_xet]"
hf auth login
```

## 3. Create the repository (once)

```
hf repos create {repo_id} --repo-type dataset --private
```

Make it public in its settings once you have checked it. To let people in only after they accept terms, turn on
access requests there (gating applies to the whole repository). Then add it to the BoneHub
[Harmonized Databases](https://huggingface.co/collections/BoneHub/harmonized-databases-6af58ac969228185e158c202) collection.

## 4. Upload

```
python -m bonehub_format upload "{root}" --repo-id {repo_id} --commit-message "Upload {title}"
```

This leaves out what `.gitignore` lists: this file, the logs, the per-label model folders (their zips are
uploaded) and every file whose license is restricted or not known. Do not upload with `hf upload` alone: it does not read `.gitignore`,
so it would upload all of it, restricted data included. Running the command again resumes an interrupted upload and
skips files that are already there.

About {size / 1e9:.1f} GB will be uploaded. Free accounts get public storage on a best-effort basis; for this much
data, plan for a PRO or organisation plan, or ask Hugging Face for a research grant (datasets@huggingface.co).

## 5. Tag the release

```
hf repos tag create {repo_id} v1.0 --repo-type dataset
```

## 6. Publish the webpage

See `github-webpage/README.md`.
"""


def write_hub_files(
    root: Path, repo_id: str, title: str, description: str | None = None, homepage: str | None = None
) -> list[Path]:
    """Write the dataset card, LICENSE.md, .gitattributes, .gitignore and upload.md of a database; return their paths.

    repo_id: the Hugging Face repository, e.g. 'BoneHub/<database>'.
    title: the name of the database.
    description: the opening paragraph of the card (Markdown); a generic one by default.
    homepage: the database's webpage, linked from the card.
    """
    root_path = Path(root)
    missing = [name for name in INDEX_FILES.values() if not (root_path / layout.INDEX_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"The card shows the index, which is missing ({', '.join(missing)}); run build_index first.")
    reader = Database(root_path)
    summary = _Summary(reader)
    files = {
        "README.md": _card(reader, summary, repo_id, title, description, homepage),
        "LICENSE.md": _license_file(summary),
        ".gitattributes": _GITATTRIBUTES,
        ".gitignore": _gitignore(reader),
        "upload.md": _upload_guide(root_path.resolve(), summary, repo_id, title),
    }
    written = []
    for name, content in files.items():
        path = root_path / name
        path.write_text(content, encoding="utf-8", newline="\n")
        written.append(path)
    return written
