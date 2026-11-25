"""Write data.js, the data this webpage shows, from the index of the database.

    python update_data.py                 # the index in ../index, as when this folder is inside the database
    python update_data.py path/to/index

Needs only pyarrow (pip install pyarrow). structures.json, written next to this file by bonehub_format.webpage, says
which groups of the body (skeleton figure, matrix rows, soft tissue) each label covers.
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
# The BoneHub registry's Dataset Browser, filled in by build_webpage; each dataset links to its entry there.
REGISTRY_PAGE_URL = "{{REGISTRY_PAGE_URL}}"
LICENSE_KINDS = ("image", "segmentation", "mesh", "nurbs")

# Index column -> column of the table on the page
DATASET_COLUMNS = {
    "dataset_id": "Dataset ID",
    "name": "Dataset Name",
    "subjects": "Subjects",
    "scans": "Scans",
    "scans_provided": "Scans With Its Data",
    "license_image": "Image License",
    "license_segmentation": "Segmentation License",
    "url": "URL",
    "paper": "Paper",
}
SCAN_COLUMNS = {
    "dataset_id": "Dataset ID",
    "subject_id": "Subject ID",
    "scan_id": "Scan ID",
    "age": "Age (years)",
    "gender": "Gender",
    "weight": "Weight (kg)",
    "height": "Height (cm)",
    "bmi": "BMI",
    "postmortem": "Postmortem",
    "modality": "Imaging Modality",
    "image": "Image Available",
    "image_published": "Image Published",
    "segmentation": "Segmentation Available",
    "mesh": "Mesh Available",
    "nurbs": "NURBS Available",
    "truncated": "Touching Image Border",
    "providers": "Provider Datasets",
    "image_license": "Image License",
    "remarks": "Remarks",
}
ORIGIN_KINDS = ("source", "manual", "model", "derived")


def _cell(value):
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return value


def _rows(records: list[dict], columns: dict) -> list[dict]:
    return [{label: _cell(row.get(column)) for column, label in columns.items()} for row in records]


def _origin_kind(origin: str) -> str:
    return origin.split(":", 1)[0] if origin else "source"


def _structures() -> dict | None:
    path = HERE / "structures.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _coverage(labels: list[dict], structures: dict | None, scan_index: dict[str, int]):
    """Scans per group (any provider, per provider, per kind, per origin) and the groups of each scan."""
    if structures is None:
        return None, {}
    covers = structures["labels"]
    scans_of = defaultdict(set)  # group -> scans
    by_provider = defaultdict(lambda: defaultdict(set))  # group -> provider -> scans
    by_kind = defaultdict(lambda: defaultdict(set))
    by_origin = defaultdict(lambda: defaultdict(set))
    groups_of_scan = defaultdict(set)
    for row in labels:
        scan = row["scan"]
        for group in covers.get(row["label"], ()):
            scans_of[group].add(scan)
            by_provider[group][row["provider"]].add(scan)
            by_kind[group][row["kind"]].add(scan)
            by_origin[group][_origin_kind(row["origin"])].add(scan)
            groups_of_scan[scan].add(group)
    coverage = {
        group: {
            "scans": len(scans),
            "providers": {str(p): len(s) for p, s in sorted(by_provider[group].items())},
            "kinds": {k: len(s) for k, s in sorted(by_kind[group].items())},
            "origins": {o: len(s) for o, s in sorted(by_origin[group].items())},
        }
        for group, scans in scans_of.items()
    }
    order = {group["id"]: i for i, group in enumerate(structures["groups"])}
    scan_groups = {scan: sorted(order[g] for g in found) for scan, found in groups_of_scan.items() if scan in scan_index}
    return coverage, scan_groups


def _dataset_info(datasets: list[dict], scans: list[dict], labels: list[dict]) -> list[dict]:
    """What a dataset card shows."""
    modalities = defaultdict(Counter)
    images = defaultdict(Counter)  # provider -> published / withheld images
    for scan in scans:
        modalities[scan["dataset_id"]][scan["modality"] or "unknown"] += 1
        if scan["image"] and scan["image_provider"] is not None:
            images[scan["image_provider"]]["published" if scan["image_published"] else "withheld"] += 1
    files = defaultdict(lambda: defaultdict(lambda: {"published": set(), "withheld": set()}))
    scans_with_labels = defaultdict(set)
    for row in labels:
        files[row["provider"]][row["kind"]]["published" if row["published"] else "withheld"].add(row["scan"])
        scans_with_labels[row["provider"]].add(row["scan"])
    cards = []
    for row in datasets:
        dataset_id = row["dataset_id"]
        kinds = {}
        for kind in LICENSE_KINDS:
            license = row.get(f"license_{kind}")
            if kind == "image":
                published, withheld = images[dataset_id]["published"], images[dataset_id]["withheld"]
            else:
                published = len(files[dataset_id][kind]["published"])
                withheld = len(files[dataset_id][kind]["withheld"])
            if license is not None or published or withheld:
                kinds[kind] = {"license": license, "published": published, "withheld": withheld}
        cards.append(
            {
                "id": dataset_id,
                "name": row["name"],
                "description": row.get("description"),
                "url": row.get("url"),
                "paper": row.get("paper"),
                "country": row.get("country"),
                "year": row.get("release_date"),
                "version": row.get("version"),
                "remarks": row.get("remarks"),
                "subjects": row["subjects"],
                "scans": row["scans"],
                "scansProvided": row["scans_provided"],
                "scansWithLabels": len(scans_with_labels[dataset_id]),
                "modalities": dict(modalities[dataset_id].most_common()),
                "kinds": kinds,
                "subsets": json.loads(row["subsets"]) if row.get("subsets") else None,
                "derivedFrom": row.get("derived_from"),
                "annotatedLabels": len(row.get("annotated_labels") or []),
                "registry": f"{REGISTRY_PAGE_URL}#dataset={dataset_id}",
            }
        )
    return cards


def write_data_js(index_dir: Path, output_path: Path) -> tuple[int, int]:
    """Write data.js from the index; return the number of datasets and scans."""
    index_dir = Path(index_dir)
    datasets = pq.read_table(index_dir / "datasets.parquet").to_pylist()
    scans = pq.read_table(index_dir / "scans.parquet").to_pylist()
    labels = pq.read_table(index_dir / "labels.parquet").to_pylist()
    structures = _structures()

    dataset_rows = _rows(datasets, DATASET_COLUMNS)
    for dataset in dataset_rows:
        dataset["Registry"] = f"{REGISTRY_PAGE_URL}#dataset={dataset['Dataset ID']}"
    scan_index = {scan["scan"]: i for i, scan in enumerate(scans)}
    coverage, scan_groups = _coverage(labels, structures, scan_index)
    scan_rows = _rows(scans, SCAN_COLUMNS)
    for scan, row in zip(scans, scan_rows):
        row["_groups"] = scan_groups.get(scan["scan"], [])  # indices into structuresData.groups; not a column
        row["_providers"] = scan["providers"] or []
    content = (
        "// Generated from the database's index by update_data.py; do not edit.\n"
        f"// Updated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC\n\n"
        f"const datasetsData = {json.dumps(dataset_rows, ensure_ascii=False)};\n"
        f"const scansData = {json.dumps(scan_rows, ensure_ascii=False, separators=(',', ':'))};\n"
        f"const datasetCards = {json.dumps(_dataset_info(datasets, scans, labels), ensure_ascii=False)};\n"
        f"const structuresData = {json.dumps(structures, ensure_ascii=False, separators=(',', ':'))};\n"
        f"const coverageData = {json.dumps(coverage, ensure_ascii=False, separators=(',', ':'))};\n"
    )
    Path(output_path).write_text(content, encoding="utf-8", newline="\n")
    return len(datasets), len(scans)


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
