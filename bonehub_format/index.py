"""The index: Parquet tables that summarise a database, to search it without opening every file.

    index/datasets.parquet   one row per dataset
    index/scans.parquet      one row per scan, with the fields of its subject
    index/labels.parquet     one row per scan, label and kind (segmentation, mesh, nurbs)

The index is generated: build_index() rewrites it from the metadata files, the headers of the images and
segmentations. It also holds what the metadata files leave out because the data says it: image size
and voxel spacing, and which segmented bones touch the image border (so are probably cut off by the field of view).

Requires pyarrow: pip install "bonehub-format[converter]".
"""

from __future__ import annotations

import gzip
import os
import struct
import tempfile
from pathlib import Path

from . import layout
from .licenses import is_restricted
from .metadata import LABEL_KINDS
from .reader import Database
from .segmentation_file import read_segmentation_segments

INDEX_FILES = {"datasets": "datasets.parquet", "scans": "scans.parquet", "labels": "labels.parquet"}


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


def collect_index(root: Path) -> dict[str, list[dict]]:
    """The rows of the index tables."""
    root = Database(root)
    datasets, scans, labels = [], [], []
    all_scans = list(root.scans())

    for dataset in root.datasets():
        info = dataset.info
        dataset_scans = [scan for scan in all_scans if scan.dataset.id == dataset.id]
        datasets.append(
            {
                "dataset_id": dataset.id,
                **{field: getattr(info, field) for field in ("name", "description", "url", "paper", "citation")},
                **{field: getattr(info, field) for field in ("country", "release_date", "version", "license")},
                "restricted": is_restricted(info.license),
                "derived_from": info.derived_from,
                "annotated_labels": info.annotated_labels,
                "remarks": info.remarks,
                "schema_version": info.schema_version,
                "subjects": len({scan.subject.id for scan in dataset_scans}),
                "scans": len(dataset_scans),
            }
        )

    for scan in all_scans:
        info, subject_info = scan.info, scan.subject.info
        geometry = nifti_geometry(scan.image_path) if scan.image_path else None
        truncated = None
        if scan.segmentation_path:
            try:
                truncated = truncated_labels(scan.segmentation_path, geometry[0] if geometry else None)
            except (OSError, ValueError):
                truncated = None
        row_ids = {"scan": scan.key, "dataset_id": scan.dataset.id, "subject_id": scan.subject.id, "scan_id": scan.id}
        scans.append(
            {
                **row_ids,
                "source_subject_id": subject_info.source_subject_id,
                "gender": subject_info.gender,
                "source_scan_id": info.source_scan_id,
                **{field: getattr(info, field) for field in ("modality", "age", "weight", "height", "bmi", "postmortem")},
                **{field: getattr(info, field) for field in ("days_from_baseline", "registered_to", "source_split")},
                "license": scan.license,
                "restricted": is_restricted(scan.license),
                "image": scan.image_path is not None,
                "image_size": geometry[0] if geometry else None,
                "image_spacing": geometry[1] if geometry else None,
                **{kind: info.labels(kind) for kind in LABEL_KINDS},
                "truncated": truncated,
                "subject_remarks": subject_info.remarks,
                "remarks": info.remarks,
            }
        )
        for kind in LABEL_KINDS:
            for label, origin in (getattr(info, kind) or {}).items():
                labels.append(
                    {
                        **row_ids,
                        "label": label,
                        "kind": kind,
                        "origin": origin,
                        "truncated": (label in truncated) if kind == "segmentation" and truncated is not None else None,
                    }
                )
    return {"datasets": datasets, "scans": scans, "labels": labels}


def _schemas(pa) -> dict:
    ids = [("dataset_id", pa.int32()), ("subject_id", pa.int32()), ("scan_id", pa.int32())]
    text, labels = pa.string(), pa.list_(pa.string())
    return {
        "datasets": pa.schema(
            [("dataset_id", pa.int32())]
            + [(field, text) for field in ("name", "description", "url", "paper", "citation", "country")]
            + [(field, text) for field in ("release_date", "version", "license")]
            + [("restricted", pa.bool_()), ("derived_from", pa.list_(pa.int32())), ("annotated_labels", labels)]
            + [("remarks", text), ("schema_version", text), ("subjects", pa.int32()), ("scans", pa.int32())]
        ),
        "scans": pa.schema(
            [("scan", text)]
            + ids
            + [("source_subject_id", text), ("gender", text), ("source_scan_id", text), ("modality", text)]
            + [("age", pa.int32()), ("weight", pa.float64()), ("height", pa.float64()), ("bmi", pa.float64())]
            + [("postmortem", pa.bool_()), ("days_from_baseline", pa.int32()), ("registered_to", text)]
            + [("source_split", text), ("license", text), ("restricted", pa.bool_()), ("image", pa.bool_())]
            + [("image_size", pa.list_(pa.int32())), ("image_spacing", pa.list_(pa.float64()))]
            + [(kind, labels) for kind in LABEL_KINDS]
            + [("truncated", labels)]
            + [("subject_remarks", text), ("remarks", text)]
        ),
        "labels": pa.schema(
            [("scan", text)] + ids + [("label", text), ("kind", text), ("origin", text), ("truncated", pa.bool_())]
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
