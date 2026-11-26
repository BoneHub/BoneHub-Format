"""Write data.js, the data this webpage shows, from the index of the database.

    python update_data.py                 # the index in ../index, as when this folder is inside the database
    python update_data.py path/to/index

Needs only pyarrow (pip install pyarrow). structures.json, written next to this file by bonehub_format.webpage, gives
the section of the body (Head, Spine, ..., Muscles) of every label.

data.js holds one object, `site`. Its scans are arrays, to keep the file small:
    [scan path, dataset index, sex, age, modality, segmented label indices, files]
and each file is [kind ('image', 'segmentation', 'mesh', 'nurbs'), number, size in bytes, published, source, license].
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
# Filled in by build_webpage: the database's Hugging Face repository and the BoneHub registry's Dataset Browser.
HF_REPO_ID = "{{HF_REPO_ID}}"
REGISTRY_PAGE_URL = "{{REGISTRY_PAGE_URL}}"
LICENSE_KINDS = ("image", "segmentation", "mesh", "nurbs")


def _table(index_dir: Path, name: str) -> list[dict]:
    return pq.read_table(index_dir / f"{name}.parquet").to_pylist()


def site_data(index_dir: Path) -> dict:
    index_dir = Path(index_dir)
    datasets, sources = _table(index_dir, "datasets"), _table(index_dir, "sources")
    scans, files = _table(index_dir, "scans"), _table(index_dir, "files")
    structures = json.loads((HERE / "structures.json").read_text(encoding="utf-8"))

    label_order = {label: i for i, label in enumerate(structures["labels"])}
    present = {label for scan in scans for label in scan["segmentation"] or []}
    labels = sorted(present, key=lambda label: label_order.get(label, len(label_order)))
    label_index = {label: i for i, label in enumerate(labels)}

    source_info = {}
    sources_of = defaultdict(list)
    for row in sources:
        sources_of[row["dataset_id"]].append([row["source_id"], row["name"], row["url"]])
        source_info.setdefault(
            row["source_id"],
            {"name": row["name"], "url": row["url"], "licenses": {kind: row[f"license_{kind}"] for kind in LICENSE_KINDS}},
        )
    dataset_index = {row["dataset_id"]: i for i, row in enumerate(datasets)}
    files_of = defaultdict(list)
    for row in files:
        files_of[row["scan"]].append([row["kind"], row["number"], row["size"], row["published"], row["source"], row["license"]])

    return {
        "updated": f"{datetime.now(timezone.utc):%Y-%m-%d}",
        "repo": HF_REPO_ID,
        "registry": REGISTRY_PAGE_URL,
        "sections": structures["sections"],
        "labels": labels,
        "labelSection": [structures["labels"].get(label, len(structures["sections"]) - 1) for label in labels],
        "datasets": [
            {
                "id": row["dataset_id"],
                "folder": f"ds-{row['dataset_id']:04d}",
                "name": row["name"],
                "subjects": row["subjects"],
                "scans": row["scans"],
                "sources": sources_of[row["dataset_id"]],
            }
            for row in datasets
        ],
        "sources": {str(key): value for key, value in sorted(source_info.items())},
        "scans": [
            [
                scan["scan"],
                dataset_index[scan["dataset_id"]],
                scan["gender"],
                scan["age"],
                scan["modality"],
                [label_index[label] for label in scan["segmentation"] or []],
                files_of[scan["scan"]],
            ]
            for scan in scans
        ],
    }


def write_data_js(index_dir: Path, output_path: Path) -> tuple[int, int]:
    """Write data.js from the index; return the number of datasets and scans."""
    data = site_data(index_dir)
    content = (
        "// Generated from the database's index by update_data.py; do not edit.\n"
        f"const site = {json.dumps(data, ensure_ascii=False, separators=(',', ':'))};\n"
    )
    Path(output_path).write_text(content, encoding="utf-8", newline="\n")
    return len(data["datasets"]), len(data["scans"])


def main(argv: list[str]) -> int:
    index_dir = Path(argv[0]) if argv else HERE.parent / "index"
    if not (index_dir / "scans.parquet").is_file():
        print(f"No index in '{index_dir}'. Build it first (bonehub_format.index.build_index).")
        return 1
    datasets, scans = write_data_js(index_dir, HERE / "data.js")
    print(f"Wrote {HERE / 'data.js'}: {datasets} datasets, {scans} scans.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
