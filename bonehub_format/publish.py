"""The files that make a database a Hugging Face dataset repository.

    README.md        the dataset card: what the data is, how it is laid out and how to download parts of it
    LICENSE.md       the licenses of each source per kind of data, and of the subsets and files that have their own
    .gitattributes   which files Hugging Face stores as large files
    .gitignore       what stays out of the upload: upload.md, logs, the per-label model folders (their zips are
                     uploaded) and every file whose license is restricted or not known
    upload.md        how to upload the database; it is not uploaded itself

Whether a file is published is decided per file, from its license (see reader.DataFile.license): a scan can publish
its segmentation and keep its restricted image back. The JSON files are always published.

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
    """(file name, kind, source, license) of every data file of a scan that would be published: the image, the
    segmentations and the model zips."""
    return [(data.path.name, data.kind, data.source, data.license()) for data in scan.files()]


def _withheld(root: Database) -> list[str]:
    """The data files that are not published (restricted or unknown license), as paths relative to the root; a file
    name withheld in every scan of a dataset is given as one pattern, e.g. ds-0003/*/*/image.nii.gz."""
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
        *((f"**/{kind}-*/", f"*/{kind}-*/*") for kind in layout.MODEL_KINDS),
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
        if pattern == "**/mesh-*/":
            lines.append("# Per-label model files: their zips (mesh-N.zip, nurbs-N.zip) are uploaded instead")
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
        self.modalities = Counter(scan.info.modality or "unknown" for scan in self.scans)
        self.kind_counts = {kind: sum(len(scan.labels(kind)) for scan in self.scans) for kind in LABEL_KINDS}
        # (dataset ID, source, kind) -> files, files withheld
        self.files = Counter()
        self.withheld = Counter()
        self.file_licenses = Counter()  # (dataset ID, source, kind, license) of files under a subset's or their own
        for scan in self.scans:
            for data in scan.files():
                key = (scan.dataset.id, data.source, data.kind)
                license = data.license()
                self.files[key] += 1
                if not is_publishable(license):
                    self.withheld[key] += 1
                if data.info.license is not None or data.info.subset is not None:
                    self.file_licenses[(*key, license)] += 1
        self.restricted = [
            (dataset, source_id, source)
            for dataset in self.datasets
            for source_id, source in dataset.info.sources.items()
            if any(self.withheld[(dataset.id, source_id, kind)] for kind in LICENSE_KINDS)
        ]

    def withheld_text(self, dataset_id: int, source_id: int) -> str:
        counts = [(kind, self.withheld[(dataset_id, source_id, kind)]) for kind in LICENSE_KINDS]
        return ", ".join(f"{count} {kind}" for kind, count in counts if count)


def _licenses_text(source) -> str:
    """A source's licenses per kind, in one cell: one license if all kinds share it."""
    if source.license is None:
        return "not given"
    given = {kind: source.license.of(kind) for kind in LICENSE_KINDS if source.license.of(kind) is not None}
    if len(set(given.values())) == 1:
        return _license_link(next(iter(given.values())))
    return "; ".join(f"{kind}: {_license_link(value)}" for kind, value in given.items()) or "not given"


def _sources_text(info) -> str:
    """The sources of a dataset, each linked to its registry entry."""
    return "<br>".join(f"[{source_id}]({registry_entry_url(source_id)}) {source.name}" for source_id, source in info.sources.items())


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
        *(line for name in ("files", "labels", "datasets", "sources")
          for line in (f"- config_name: {name}", f"  data_files: {layout.INDEX_DIR}/{INDEX_FILES[name]}")),
        "---",
    ]

    subjects = sum(summary.subjects_by_dataset.values())
    modalities = ", ".join(f"{count} {modality}" for modality, count in summary.modalities.most_common())
    kinds = ", ".join(f"{count} {kind} labels" for kind, count in summary.kind_counts.items() if count)
    rows = [
        "| Dataset | Name | Sources (BoneHub registry) | Subjects | Scans | Licenses |",
        "|---|---|---|---|---|---|",
    ]
    for dataset in summary.datasets:
        info = dataset.info
        licenses = "<br>".join(f"{source_id}: {_licenses_text(source)}" for source_id, source in info.sources.items())
        rows.append(
            f"| {dataset.path.name} | {info.name} | {_sources_text(info)} | {summary.subjects_by_dataset[dataset.id]} "
            f"| {summary.by_dataset[dataset.id]} | {licenses} |"
        )
    sources = sorted({source_id for dataset in summary.datasets for source_id in dataset.info.sources})

    body = [
        f"# {title}",
        "",
        description or f"3D bone data in the [BoneHub format]({FORMAT_URL}): medical images, bone segmentations and "
        "the meshes and NURBS surfaces derived from them, converted from public datasets.",
        "",
        f"{len(summary.datasets)} datasets made from {len(sources)} source datasets of the [BoneHub registry]"
        f"({REGISTRY_PAGE_URL}), with {subjects} subjects and {len(summary.scans)} scans ({modalities}); {kinds}."
        + (f" Website: {homepage}" if homepage else ""),
        "",
        "## Datasets",
        "",
        *rows,
        "",
    ]
    if summary.restricted:
        lines = [f"- {dataset.path.name}, source {source_id} ({source.name}): {summary.withheld_text(dataset.id, source_id)} files"
                 for dataset, source_id, source in summary.restricted]
        body += [
            "Some files are under a restricted license (or none is known), so they are listed in the index but not in "
            "this repository; the `published` column of `index/files.parquet` says which. Obtain them from their "
            "source and convert them with the BoneHub-Format package:",
            "",
            *lines,
            "",
        ]
    body += [
        "## Layout",
        "",
        "```",
        "ds-0001/                         a dataset, numbered within this database",
        "├── dataset.json                 its sources (registry datasets): names, links, licenses, labels covered",
        "└── sub-000001/                  a subject (person)",
        "    ├── subject.json             gender, the subject's ID in each source",
        "    └── scan-01/                 a scan of the subject",
        "        ├── scan.json            modality, age, weight, height, ...",
        "        ├── image.nii.gz         + image.json           the source of the image, its ID there",
        "        ├── segmentation-1.nrrd  + segmentation-1.json  its source, labels and their origin",
        "        ├── segmentation-2.nrrd  + segmentation-2.json  another source's segmentation of the same image",
        "        ├── mesh-1.zip           + mesh-1.json          <LABEL>.stl",
        "        └── nurbs-1.zip          + nurbs-1.json         <LABEL>.iges / <LABEL>.step",
        "index/                           Parquet tables of the datasets, sources, scans, files and labels",
        "```",
        "",
        "A folder's name is its ID. The index (also shown in the dataset viewer above) holds every field of the "
        "metadata files, plus image size and voxel spacing, the structures that touch the image border, and the "
        "path, size and license of every file.",
        "",
        "## Downloading",
        "",
        "Everything (published files only):",
        "",
        "```",
        f"hf download {repo_id} --repo-type dataset --local-dir data",
        "```",
        "",
        "One dataset, e.g. the images and segmentations of ds-0001:",
        "",
        "```",
        f'hf download {repo_id} --repo-type dataset --local-dir data --include "ds-0001/*.json" '
        '--include "ds-0001/*/*.json" --include "ds-0001/*/*/*.json" --include "ds-0001/*/*/image.nii.gz" '
        '--include "ds-0001/*/*/segmentation-*.nrrd"',
        "```",
        "",
        "Or choose scans in the index and download only their files:",
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
        "Single files can be downloaded from the browser too: every file has a direct link, "
        f"`https://huggingface.co/datasets/{repo_id}/resolve/main/<path>?download=true`, with `<path>` from the `path` "
        "column of `index/files.parquet` (the database's webpage lists them per scan).",
        "",
        "## Reading the data with Python",
        "",
        f"`pip install bonehub-format @ git+{FORMAT_URL}.git`",
        "",
        "```python",
        "from bonehub_format import Database, read_segmentation",
        "",
        'for scan in Database("data").scans():',
        "    print(scan.key, scan.info.modality, scan.info.age, scan.image_path)",
        '    for segmentation in scan.files("segmentation"):',
        "        mask = read_segmentation(segmentation.path)   # SimpleITK image of LabelMap values",
        "        print(segmentation.source, segmentation.labels, segmentation.license())",
        '    for mesh in scan.files("mesh"):',
        "        for label, files in mesh.model_files().items():   # read from the zip",
        "            stl_bytes = files[0].read_bytes()",
        "```",
        "",
        "Without the package, the files are plain NIfTI, NRRD, STL, IGES and STEP files and JSON: the voxels of a "
        "segmentation hold segment numbers, which its NRRD header maps to label names (`Segment<i>_Name`) and LabelMap "
        "values (`Segment<i>_LabelValue`).",
        "",
        "## Conventions",
        "",
        f"- **Labels** are the names of the [BoneHub label map]({LABEL_MAP_URL}).",
        "- **Sources**: a dataset is made from one or more datasets of the "
        f"[BoneHub registry]({REGISTRY_PAGE_URL}), its sources, listed in its `dataset.json` by their BoneHub ID. "
        "Every data file has a JSON file of the same name that names its source and the scan's ID there.",
        "- **Several versions**: where two sources segmented the same structure of a scan, both are kept, as "
        "`segmentation-1.nrrd` and `segmentation-2.nrrd`; the files of a kind are numbered within the scan.",
        "- **Origin**: every label says how its source came by it: `source` (with the source dataset), "
        "`manual`, `model:<name>@<version>` or `derived:<tool>@<version>` (made from another file of the scan, "
        "e.g. a mesh from the segmentation).",
        "- **Coordinates**: meshes and NURBS surfaces are in the physical space of the image, in LPS and millimetres.",
        "- **Coverage**: `annotated_labels` of a source in `dataset.json` lists the labels its segmentations cover. In a "
        "scan with a segmentation of that source, a covered label that is missing is absent from the image; other "
        "labels were not segmented.",
        "- **One subject, one person**: each person is stored once. Where several sources hold the same person or "
        "image, their data is merged into one subject, whose `subject.json` gives its ID in each of them.",
        "",
        "## Licenses",
        "",
        "Each file keeps the license of its source, for its kind of data (image, segmentation, mesh, NURBS), or of the "
        "source's collection it belongs to: see [LICENSE.md](LICENSE.md), and the `license` column of "
        "`index/files.parquet`. Cite the source datasets you use; their registry entries link their papers.",
        "",
    ]
    return "\n".join(yaml + body)


def _license_file(summary: _Summary) -> str:
    lines = [
        "# Licenses",
        "",
        "Each file is under the license of its source (a dataset of the BoneHub registry, named in the file's JSON), "
        "for its kind of data: `image.nii.gz` under the source's image license, `segmentation-N.nrrd` under its "
        "segmentation license, `mesh-N.zip` and `nurbs-N.zip` under its mesh or NURBS license. Collections of a source "
        "with terms of their own, and files with a license of their own, are listed below the table; "
        "`index/files.parquet` (`license`) gives the license of every file.",
        "",
        "| Dataset | Source | Name | Image | Segmentation | Mesh | NURBS | Link |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for dataset in summary.datasets:
        for source_id, source in dataset.info.sources.items():
            cells = [_license_link(source.license.of(kind)) if source.license else "not given" for kind in LICENSE_KINDS]
            lines.append(
                f"| {dataset.path.name} | [{source_id}]({registry_entry_url(source_id)}) | {source.name} | {' | '.join(cells)} "
                f"| {source.url or ''} |"
            )
    subsets = [
        f"| {source_id} | {name} | {subset.name} | "
        + " | ".join(_license_link(subset.license.of(kind)) if subset.license and subset.license.of(kind) else "as the source" for kind in LICENSE_KINDS)
        + f" | {subset.url or ''} |"
        for source_id, source in sorted({sid: src for d in summary.datasets for sid, src in d.info.sources.items()}.items())
        for name, subset in (source.subsets or {}).items()
    ]
    if subsets:
        lines += [
            "",
            "## Collections with licenses of their own",
            "",
            "| Source | Collection | Name | Image | Segmentation | Mesh | NURBS | Link |",
            "|---|---|---|---|---|---|---|---|",
            *subsets,
        ]
    overrides = [
        f"- {layout.dataset_dir_name(dataset_id)}, source {source_id}, {kind}: {count} files under {_license_link(value)}"
        + ("" if is_publishable(value) else " (not published)")
        for (dataset_id, source_id, kind, value), count in sorted(summary.file_licenses.items(), key=lambda item: tuple(map(str, item[0])))
    ]
    if overrides:
        lines += ["", "## Files under the license of their collection or their own", "", *overrides]
    if summary.restricted:
        lines += [
            "",
            "## Files that are not published",
            "",
            "These files are under a custom license with restricted access, or no license is known for them. They are "
            "listed in the index, but they are not in this repository; obtain them from their source.",
            "",
        ]
        for dataset, source_id, source in summary.restricted:
            lines.append(
                f"- {dataset.path.name}, source {source_id} ({source.name}): {summary.withheld_text(dataset.id, source_id)} "
                f"files ({source.url or 'no URL given'})"
            )
    return "\n".join(lines) + "\n"


def _upload_guide(root: Path, summary: _Summary, repo_id: str, title: str) -> str:
    size = 0
    for scan in summary.scans:
        size += (scan.path / layout.SCAN_JSON).stat().st_size
        for data in scan.files():
            size += data.json_path.stat().st_size
            if is_publishable(data.license()) and data.path.is_file():
                size += data.path.stat().st_size
    return f"""# Uploading {title} to Hugging Face

This file is listed in `.gitignore`, so it is not uploaded.

## 1. Prepare the database

```
python -m bonehub_format prepare "{root}" --repo-id {repo_id} --title "{title}"
```

This brings the zips of the model folders (`mesh-N/`, `nurbs-N/`) up to date, rebuilds the index, checks the database, and writes
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
