"""Write the data this webpage shows, from the index of the database: data.js and details/<dataset>.js.

    python update_data.py                 # the index in ../index, as when this folder is inside the database
    python update_data.py path/to/index

Needs only pyarrow (pip install pyarrow). structures.json, written next to this file by bonehub_format.webpage, gives
the section of the body (Head, Spine, ..., Muscles) of every label.

data.js holds `site`: what the overview, the scan table and its filters need, one scan per line. A scan's labels are
indices into `site.labels`. details/ds-NNNN.js holds what the data card of each scan of a dataset shows (the dataset
and its sources, the subject's IDs in its sources, image geometry, the labels of each file with their origin); the
page loads it when a card of that dataset is opened.
"""

import json
import shutil
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
LABELED_KINDS = ("segmentation", "mesh", "nurbs")
DETAILS_DIR = "details"


def _table(index_dir: Path, name: str) -> list[dict]:
    return pq.read_table(index_dir / f"{name}.parquet").to_pylist()


def _compact(record: dict) -> dict:
    """The fields that have a value: no None, no empty list or dict."""
    return {key: value for key, value in record.items() if value is not None and value != [] and value != {}}


def _line(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def _js(value, indent: int = 0) -> str:
    """JSON with one entry per line for the containers of records (lists and maps of dicts), and every record or
    flat list on a line of its own."""
    pad, inner = " " * indent, " " * (indent + 2)
    if isinstance(value, dict) and value and any(isinstance(v, (dict, list)) for v in value.values()):
        items = [f"{inner}{_line(key)}: {_js(item, indent + 2)}" for key, item in value.items()]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        return "[\n" + ",\n".join(f"{inner}{_line(item)}" for item in value) + f"\n{pad}]"
    return _line(value)


def _records(value) -> str:
    """A map of records, one record per line."""
    return "{\n" + ",\n".join(f"  {_line(key)}: {_line(item)}" for key, item in value.items()) + "\n}"


def site_data(index_dir: Path) -> tuple[dict, dict[str, dict]]:
    """`site` for data.js, and the details of each dataset folder for details/<folder>.js."""
    index_dir = Path(index_dir)
    datasets, sources = _table(index_dir, "datasets"), _table(index_dir, "sources")
    scans, files, labels = _table(index_dir, "scans"), _table(index_dir, "files"), _table(index_dir, "labels")
    structures = json.loads((HERE / "structures.json").read_text(encoding="utf-8"))

    label_order = {label: i for i, label in enumerate(structures["labels"])}
    present = {label for scan in scans for kind in LABELED_KINDS for label in scan[kind] or []}
    names = sorted(present, key=lambda label: label_order.get(label, len(label_order)))
    label_index = {label: i for i, label in enumerate(names)}
    other_section = len(structures["sections"]) - 1

    folder = {row["dataset_id"]: f"ds-{row['dataset_id']:04d}" for row in datasets}
    dataset_index = {row["dataset_id"]: i for i, row in enumerate(datasets)}
    sources_of = defaultdict(list)
    for row in sources:
        sources_of[row["dataset_id"]].append(row)
    files_of = defaultdict(list)
    for row in files:
        files_of[row["scan"]].append(row)
    origins_of = defaultdict(lambda: defaultdict(list))  # (scan, kind, number) -> origin -> label indices
    for row in labels:
        origins_of[(row["scan"], row["kind"], row["number"])][row["origin"]].append(label_index[row["label"]])

    site = {
        "updated": f"{datetime.now(timezone.utc):%Y-%m-%d}",
        "repo": HF_REPO_ID,
        "registry": REGISTRY_PAGE_URL,
        "sections": structures["sections"],
        "labels": [{"name": name, "section": structures["labels"].get(name, other_section)} for name in names],
        "labelmap": {key: structures[key] for key in ("fields", "nodes", "tissueBroader")},
        "datasets": [
            {
                "id": row["dataset_id"],
                "folder": folder[row["dataset_id"]],
                "name": row["name"],
                "subjects": row["subjects"],
                "scans": row["scans"],
                "sources": [[source["source_id"], source["name"]] for source in sources_of[row["dataset_id"]]],
            }
            for row in datasets
        ],
        "sources": {
            str(row["source_id"]): {"name": row["name"], "url": row["url"]}
            for row in sorted(sources, key=lambda row: row["source_id"])
        },
        "scans": [
            _compact(
                {
                    "scan": scan["scan"],
                    "dataset": dataset_index[scan["dataset_id"]],
                    "sex": scan["gender"],
                    "age": scan["age"],
                    "weight": scan["weight"],
                    "height": scan["height"],
                    "bmi": scan["bmi"],
                    "race": scan["race"],
                    "ethnicity": scan["ethnicity"],
                    "modality": scan["modality"],
                    **{kind: [label_index[label] for label in scan[kind] or []] for kind in LABELED_KINDS},
                    "files": [
                        _compact(
                            {
                                "kind": row["kind"],
                                "number": row["number"],
                                "size": row["size"],
                                "published": row["published"],
                                "source": row["source"],
                                "license": row["license"],
                            }
                        )
                        for row in files_of[scan["scan"]]
                    ],
                }
            )
            for scan in scans
        ],
    }

    details = {}
    for row in datasets:
        dataset_sources = [
            _compact(
                {
                    "id": source["source_id"],
                    "name": source["name"],
                    "description": source["description"],
                    "url": source["url"],
                    "paper": source["paper"],
                    "country": source["country"],
                    "release": source["release_date"],
                    "version": source["version"],
                    "licenses": _compact({kind: source[f"license_{kind}"] for kind in LICENSE_KINDS}),
                    "subsets": json.loads(source["subsets"]) if source["subsets"] else None,
                    "remarks": source["remarks"],
                }
            )
            for source in sources_of[row["dataset_id"]]
        ]
        details[folder[row["dataset_id"]]] = {
            "dataset": _compact({"name": row["name"], "description": row["description"], "remarks": row["remarks"]}),
            "sources": dataset_sources,
            "scans": {},
        }
    for scan in scans:
        details[folder[scan["dataset_id"]]]["scans"][scan["scan"]] = _compact(
            {
                "subjectSources": json.loads(scan["subject_sources"]) if scan["subject_sources"] else None,
                "subjectRemarks": scan["subject_remarks"],
                "postmortem": scan["postmortem"],
                "daysFromBaseline": scan["days_from_baseline"],
                "registeredTo": scan["registered_to"],
                "remarks": scan["remarks"],
                "imageSize": scan["image_size"],
                "imageSpacing": scan["image_spacing"],
                "truncated": [label_index[label] for label in scan["truncated"] or [] if label in label_index],
                "files": [
                    _compact(
                        {
                            "kind": row["kind"],
                            "number": row["number"],
                            "sourceScanId": row["source_scan_id"],
                            "subset": row["subset"],
                            "origins": dict(origins_of[(scan["scan"], row["kind"], row["number"])]),
                        }
                    )
                    for row in files_of[scan["scan"]]
                ],
            }
        )
    return site, details


def write_data_js(index_dir: Path, output_path: Path) -> tuple[int, int]:
    """Write data.js and details/<dataset>.js (next to data.js) from the index; return the number of datasets and
    scans."""
    site, details = site_data(index_dir)
    output_path = Path(output_path)
    header = "// Generated from the database's index by update_data.py; do not edit.\n"
    body = ",\n".join(f"  {_line(key)}: {_js(value, 2)}" for key, value in site.items())
    output_path.write_text(f"{header}const site = {{\n{body}\n}};\n", encoding="utf-8", newline="\n")
    folder = output_path.parent / DETAILS_DIR
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir()
    for name, data in details.items():
        body = ",\n".join(
            f"  {_line(key)}: {_records(value) if key == 'scans' else _js(value, 2)}" for key, value in data.items()
        )
        (folder / f"{name}.js").write_text(
            f"{header}(window.siteDetails = window.siteDetails || {{}})[{_line(name)}] = {{\n{body}\n}};\n",
            encoding="utf-8",
            newline="\n",
        )
    return len(site["datasets"]), len(site["scans"])


def main(argv: list[str]) -> int:
    index_dir = Path(argv[0]) if argv else HERE.parent / "index"
    if not (index_dir / "scans.parquet").is_file():
        print(f"No index in '{index_dir}'. Build it first (bonehub_format.index.build_index).")
        return 1
    datasets, scans = write_data_js(index_dir, HERE / "data.js")
    print(f"Wrote {HERE / 'data.js'} and {HERE / DETAILS_DIR}/: {datasets} datasets, {scans} scans.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
