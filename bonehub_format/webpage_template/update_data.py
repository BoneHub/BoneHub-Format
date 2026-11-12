"""Write data.js, the data this webpage shows, from the index of the dataset.

    python update_data.py                 # the index in ../index, as when this folder is inside the dataset
    python update_data.py path/to/index

Needs only pyarrow (pip install pyarrow).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent

# Index column -> column of the table on the page
DATASET_COLUMNS = {
    "dataset_id": "Dataset ID",
    "name": "Dataset Name",
    "subjects": "Subjects",
    "scans": "Scans",
    "license": "License",
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
    "segmentation": "Segmentation Available",
    "mesh": "Mesh Available",
    "nurbs": "NURBS Available",
    "truncated": "Touching Image Border",
    "license": "License",
    "remarks": "Remarks",
}


def _cell(value):
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return value


def _rows(path: Path, columns: dict) -> list[dict]:
    return [{label: _cell(row.get(column)) for column, label in columns.items()} for row in pq.read_table(path).to_pylist()]


def write_data_js(index_dir: Path, output_path: Path) -> tuple[int, int]:
    """Write data.js from the index; return the number of datasets and scans."""
    index_dir = Path(index_dir)
    datasets = _rows(index_dir / "datasets.parquet", DATASET_COLUMNS)
    scans = _rows(index_dir / "scans.parquet", SCAN_COLUMNS)
    content = (
        "// Generated from the dataset's index by update_data.py; do not edit.\n"
        f"// Updated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC\n\n"
        f"const datasetsData = {json.dumps(datasets, ensure_ascii=False)};\n"
        f"const scansData = {json.dumps(scans, ensure_ascii=False)};\n"
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
