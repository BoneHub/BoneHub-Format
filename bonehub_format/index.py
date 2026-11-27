"""The index: Parquet tables that summarise a database, to search it without opening every file.

    index/datasets.parquet   one row per dataset
    index/sources.parquet    one row per dataset and source (a registry dataset it is made from)
    index/scans.parquet      one row per scan, with the fields of its subject
    index/files.parquet      one row per data file (image, segmentation, mesh or NURBS zip): its path, size and license
    index/labels.parquet     one row per data file and label

The index is generated: build_index() rewrites it from the metadata files, the headers of the images and
segmentations. It also holds what the metadata files leave out because the data says it: image size and voxel
spacing, which segmented structures touch the image border (so are probably cut off by the field of view), the
license each file is under, whether it is published, and its size.

Requires pyarrow: pip install "bonehub-format[converter]".
"""

from __future__ import annotations

import gzip
import os
import struct
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import layout
from .licenses import LICENSE_KINDS, is_publishable
from .metadata import LABEL_KINDS, sort_labels
from .reader import Database, Subject
from .segmentation_file import read_segmentation_segments

#: Subjects read at once: the time is spent waiting for the disk, often a network share, not computing.
READ_THREADS = 16

INDEX_FILES = {
    "datasets": "datasets.parquet",
    "sources": "sources.parquet",
    "scans": "scans.parquet",
    "files": "files.parquet",
    "labels": "labels.parquet",
}


def _arrow():
    try:
        import pyarrow
        import pyarrow.parquet
    except ImportError as e:
        raise ImportError('Building the index needs pyarrow: pip install "bonehub-format[converter]"') from e
    return pyarrow, pyarrow.parquet


def nifti_geometry(path: Path) -> tuple[list[int], list[float]] | None:
    """Size and voxel spacing of a NIfTI-1 image, from its header; None if the header cannot be read."""
    opener = gzip.open if path.name.endswith(".gz") else open
    try:
        with opener(path, "rb") as f:
            header = f.read(348)
    except OSError:
        return None
    if len(header) < 348:
        return None
    for endian in "<>":
        if struct.unpack(f"{endian}i", header[:4])[0] == 348:
            break
    else:
        return None  # not NIfTI-1
    dim = struct.unpack(f"{endian}8h", header[40:56])
    pixdim = struct.unpack(f"{endian}8f", header[76:108])
    ndim = max(0, min(dim[0], 7))
    return list(dim[1 : 1 + ndim]), [round(float(v), 6) for v in pixdim[1 : 1 + min(ndim, 3)]]


def truncated_labels(segmentation_path: Path, image_size: list[int] | None) -> list[str] | None:
    """The labels whose segment touches the border of the image grid; None if this cannot be told from the
    headers (no extents, or a mask on another grid than the image)."""
    sizes, segments = read_segmentation_segments(segmentation_path)
    if image_size is None or list(sizes) != list(image_size[: len(sizes)]):
        return None
    truncated = []
    for label, extent in segments.items():
        if extent is None or len(extent) != 2 * len(sizes):
            return None
        if any(extent[2 * axis] <= 0 or extent[2 * axis + 1] >= sizes[axis] - 1 for axis in range(len(sizes))):
            truncated.append(label)
    return truncated


def _subject_rows(root: Path, dataset, subject_id: int) -> tuple[dict[str, list[dict]], list] | None:
    """The scan, file and label rows of one subject, and its files (data file, scan key); None if the folder is not a
    subject (no subject.json)."""
    try:
        subject = Subject(dataset, subject_id)
    except FileNotFoundError:
        return None
    rows = {"scans": [], "files": [], "labels": []}
    files_of_subject = []
    for scan in subject.scans():
        files = scan.files()  # listed once per scan
        _scan_rows(root, scan, rows, files)
        files_of_subject += [(data, scan.key) for data in files]
    return rows, files_of_subject


def collect_index(root: Path) -> dict[str, list[dict]]:
    """The rows of the index tables. The subjects are read in parallel and kept in their order."""
    root = Path(root)
    database = Database(root)
    rows = {name: [] for name in INDEX_FILES}
    for dataset in database.datasets():
        info = dataset.info
        candidates = sorted(i for i in (layout.parse_subject_dir(entry.name) for entry in dataset.path.iterdir()) if i is not None)
        with ThreadPoolExecutor(max_workers=READ_THREADS) as pool:
            subjects = [found for found in pool.map(lambda i: _subject_rows(root, dataset, i), candidates) if found is not None]
        files_of = {}  # source -> (files, scans) of the dataset that came from it
        scan_rows = [row for subject_rows, _ in subjects for row in subject_rows["scans"]]
        for _, files in subjects:
            for data, key in files:
                count, keys = files_of.setdefault(data.source, ([0], set()))
                count[0] += 1
                keys.add(key)
        rows["datasets"].append(
            {
                "dataset_id": dataset.id,
                "name": info.name,
                "description": info.description,
                "sources": list(info.sources),
                "remarks": info.remarks,
                "schema_version": info.schema_version,
                "subjects": len({row["subject_id"] for row in scan_rows}),
                "scans": len(scan_rows),
            }
        )
        for source_id, source in info.sources.items():
            count, keys = files_of.get(source_id, ([0], set()))
            rows["sources"].append(
                {
                    "dataset_id": dataset.id,
                    "source_id": source_id,
                    **{field: getattr(source, field) for field in ("name", "description", "url", "paper", "country")},
                    **{field: getattr(source, field) for field in ("release_date", "version")},
                    **{f"license_{kind}": source.license.of(kind) if source.license else None for kind in LICENSE_KINDS},
                    "subsets": json.dumps({name: subset.to_dict() for name, subset in source.subsets.items()}, ensure_ascii=False)
                    if source.subsets
                    else None,
                    "annotated_labels": source.annotated_labels,
                    "remarks": source.remarks,
                    "files": count[0],
                    "scans": len(keys),
                }
            )
        for subject_rows, _ in subjects:
            for name in ("scans", "files", "labels"):
                rows[name] += subject_rows[name]
    return rows


def _scan_rows(root: Path, scan, rows: dict[str, list[dict]], files: list) -> None:
    info, subject_info = scan.info, scan.subject.info
    image = next((data for data in files if data.kind == "image"), None)
    geometry = nifti_geometry(image.path) if image is not None and image.path.is_file() else None
    first = image or (files[0] if files else None)
    row_ids = {"scan": scan.key, "dataset_id": scan.dataset.id, "subject_id": scan.subject.id, "scan_id": scan.id}
    truncated_by_file = {}
    for data in files:
        if data.kind == "segmentation":
            try:
                truncated_by_file[data.number] = truncated_labels(data.path, geometry[0] if geometry else None)
            except (OSError, ValueError):
                truncated_by_file[data.number] = None
        license = data.license()
        rows["files"].append(
            {
                **row_ids,
                "kind": data.kind,
                "number": data.number,
                "path": data.path.relative_to(root).as_posix(),
                "source": data.source,
                "source_scan_id": data.info.source_scan_id,
                "subset": data.info.subset,
                "license": license,
                "published": is_publishable(license),
                "size": data.path.stat().st_size if data.path.is_file() else None,
                "labels": data.labels,
            }
        )
        found = truncated_by_file.get(data.number) if data.kind == "segmentation" else None
        for label, origin in getattr(data.info, "labels", {}).items():
            rows["labels"].append(
                {
                    **row_ids,
                    "kind": data.kind,
                    "number": data.number,
                    "source": data.source,
                    "subset": data.info.subset,
                    "label": label,
                    "origin": origin,
                    "license": license,
                    "published": is_publishable(license),
                    "truncated": (label in found) if found is not None else None,
                }
            )
    known = [found for found in truncated_by_file.values() if found is not None]
    image_license = image.license() if image is not None else None
    rows["scans"].append(
        {
            **row_ids,
            "source_subject_id": subject_info.source_subject_id(first.source) if first is not None else None,
            "subject_sources": json.dumps({str(k): v for k, v in subject_info.sources.items()}, ensure_ascii=False),
            "gender": subject_info.gender,
            "race": subject_info.race,
            "ethnicity": subject_info.ethnicity,
            "source_scan_id": first.info.source_scan_id if first is not None else None,
            "sources": sorted({data.source for data in files}),
            **{field: getattr(info, field) for field in ("modality", "age", "weight", "height", "bmi", "postmortem")},
            **{field: getattr(info, field) for field in ("days_from_baseline", "registered_to")},
            "source_split": first.info.source_split if first is not None else None,
            "image": image is not None,
            "image_source": image.source if image is not None else None,
            "image_license": image_license,
            "image_published": image is not None and is_publishable(image_license),
            "image_size": geometry[0] if geometry else None,
            "image_spacing": geometry[1] if geometry else None,
            **{kind: sort_labels({label for data in files if data.kind == kind for label in data.labels}) for kind in LABEL_KINDS},
            "truncated": sorted({label for found in known for label in found}) if known else None,
            "subject_remarks": subject_info.remarks,
            "remarks": info.remarks,
        }
    )


def _schemas(pa) -> dict:
    ids = [("dataset_id", pa.int32()), ("subject_id", pa.int32()), ("scan_id", pa.int32())]
    text, labels = pa.string(), pa.list_(pa.string())
    return {
        "datasets": pa.schema(
            [("dataset_id", pa.int32()), ("name", text), ("description", text), ("sources", pa.list_(pa.int32()))]
            + [("remarks", text), ("schema_version", text), ("subjects", pa.int32()), ("scans", pa.int32())]
        ),
        "sources": pa.schema(
            [("dataset_id", pa.int32()), ("source_id", pa.int32())]
            + [(field, text) for field in ("name", "description", "url", "paper", "country")]
            + [(field, text) for field in ("release_date", "version")]
            + [(f"license_{kind}", text) for kind in LICENSE_KINDS]
            + [("subsets", text), ("annotated_labels", labels), ("remarks", text)]
            + [("files", pa.int32()), ("scans", pa.int32())]
        ),
        "scans": pa.schema(
            [("scan", text)]
            + ids
            + [("source_subject_id", text), ("subject_sources", text), ("gender", text), ("race", text), ("ethnicity", text)]
            + [("source_scan_id", text), ("sources", pa.list_(pa.int32()))]
            + [("modality", text), ("age", pa.int32()), ("weight", pa.float64()), ("height", pa.float64())]
            + [("bmi", pa.float64()), ("postmortem", pa.bool_()), ("days_from_baseline", pa.int32())]
            + [("registered_to", text), ("source_split", text), ("image", pa.bool_()), ("image_source", pa.int32())]
            + [("image_license", text), ("image_published", pa.bool_())]
            + [("image_size", pa.list_(pa.int32())), ("image_spacing", pa.list_(pa.float64()))]
            + [(kind, labels) for kind in LABEL_KINDS]
            + [("truncated", labels)]
            + [("subject_remarks", text), ("remarks", text)]
        ),
        "files": pa.schema(
            [("scan", text)]
            + ids
            + [("kind", text), ("number", pa.int32()), ("path", text), ("source", pa.int32()), ("source_scan_id", text)]
            + [("subset", text), ("license", text), ("published", pa.bool_()), ("size", pa.int64()), ("labels", labels)]
        ),
        "labels": pa.schema(
            [("scan", text)]
            + ids
            + [("kind", text), ("number", pa.int32()), ("source", pa.int32()), ("subset", text), ("label", text)]
            + [("origin", text), ("license", text), ("published", pa.bool_()), ("truncated", pa.bool_())]
        ),
    }


def build_index(root: Path) -> dict[str, Path]:
    """Rewrite the index of a database; return the path of each table."""
    pa, pq = _arrow()
    root = Path(root)
    rows = collect_index(root)
    schemas = _schemas(pa)
    index_dir = root / layout.INDEX_DIR
    index_dir.mkdir(exist_ok=True)
    paths = {}
    for name, file_name in INDEX_FILES.items():
        table = pa.Table.from_pylist(rows[name], schema=schemas[name])
        path = index_dir / file_name
        fd, temp_path = tempfile.mkstemp(dir=index_dir, prefix=f".{file_name}.", suffix=".tmp")
        os.close(fd)
        try:
            pq.write_table(table, temp_path, compression="zstd")
            os.replace(temp_path, path)
        except BaseException:
            Path(temp_path).unlink(missing_ok=True)
            raise
        paths[name] = path
    return paths
