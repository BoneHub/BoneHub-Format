"""The index: Parquet tables that summarise a database, to search it without opening every file.

    index/datasets.parquet   one row per dataset
    index/scans.parquet      one row per scan, with the fields of its subject
    index/labels.parquet     one row per scan, provider dataset, label and kind (segmentation, mesh, nurbs)

The index is generated: build_index() rewrites it from the metadata files, the headers of the images and
segmentations. It also holds what the metadata files leave out because the data says it: image size and voxel
spacing, which segmented structures touch the image border (so are probably cut off by the field of view), the
license each file is under and whether it is published.

Requires pyarrow: pip install "bonehub-format[converter]".
"""

from __future__ import annotations

import gzip
import os
import struct
import tempfile
from pathlib import Path

import json

from . import layout
from .licenses import LICENSE_KINDS, is_publishable
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
    provided = {}  # dataset ID -> the scans it gives data to
    for scan in all_scans:
        for dataset_id in scan.provider_ids:
            provided.setdefault(dataset_id, set()).add(scan.key)

    for dataset in root.datasets():
        info = dataset.info
        dataset_scans = [scan for scan in all_scans if scan.dataset.id == dataset.id]
        datasets.append(
            {
                "dataset_id": dataset.id,
                **{field: getattr(info, field) for field in ("name", "description", "url", "paper", "citation")},
                **{field: getattr(info, field) for field in ("country", "release_date", "version")},
                **{f"license_{kind}": info.license.of(kind) if info.license else None for kind in LICENSE_KINDS},
                "subsets": json.dumps(
                    {name: subset.to_dict() for name, subset in info.subsets.items()}, ensure_ascii=False
                )
                if info.subsets
                else None,
                "derived_from": info.derived_from,
                "annotated_labels": info.annotated_labels,
                "remarks": info.remarks,
                "schema_version": info.schema_version,
                "subjects": len({scan.subject.id for scan in dataset_scans}),
                "scans": len(dataset_scans),
                "scans_provided": len(provided.get(dataset.id, ())),
            }
        )

    for scan in all_scans:
        info, subject_info = scan.info, scan.subject.info
        geometry = nifti_geometry(scan.image_path) if scan.image_path else None
        truncated_by_provider = {}
        for dataset_id, path in scan.segmentation_paths().items():
            try:
                truncated_by_provider[dataset_id] = truncated_labels(path, geometry[0] if geometry else None)
            except (OSError, ValueError):
                truncated_by_provider[dataset_id] = None
        known = [found for found in truncated_by_provider.values() if found is not None]
        truncated = sorted({label for found in known for label in found}) if known else None
        image_license = scan.license("image")
        image_provider = info.provider(info.image_provider) if info.image is not None else None
        row_ids = {"scan": scan.key, "dataset_id": scan.dataset.id, "subject_id": scan.subject.id, "scan_id": scan.id}
        scans.append(
            {
                **row_ids,
                "source_subject_id": subject_info.source_subject_id(scan.dataset.id),
                "gender": subject_info.gender,
                "source_scan_id": info.source_scan_id(),
                "providers": scan.provider_ids,
                "image_provider": info.image_provider,
                **{field: getattr(info, field) for field in ("modality", "age", "weight", "height", "bmi", "postmortem")},
                **{field: getattr(info, field) for field in ("days_from_baseline", "registered_to")},
                "source_split": image_provider.source_split if image_provider else None,
                "image": scan.image_path is not None,
                "image_license": image_license,
                "image_published": scan.image_path is not None and is_publishable(image_license),
                "image_size": geometry[0] if geometry else None,
                "image_spacing": geometry[1] if geometry else None,
                **{kind: info.labels(kind) for kind in LABEL_KINDS},
                "truncated": truncated,
                "subject_remarks": subject_info.remarks,
                "remarks": info.remarks,
            }
        )
        for key, provider in info.providers.items():
            dataset_id = layout.parse_provider_key(key)
            for kind in LABEL_KINDS:
                license = scan.license(kind, dataset_id)
                found = truncated_by_provider.get(dataset_id) if kind == "segmentation" else None
                for label, origin in (getattr(provider, kind) or {}).items():
                    labels.append(
                        {
                            **row_ids,
                            "provider": dataset_id,
                            "subset": provider.subset,
                            "label": label,
                            "kind": kind,
                            "origin": origin,
                            "license": license,
                            "published": is_publishable(license),
                            "truncated": (label in found) if found is not None else None,
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
            + [(field, text) for field in ("release_date", "version")]
            + [(f"license_{kind}", text) for kind in LICENSE_KINDS]
            + [("subsets", text), ("derived_from", pa.list_(pa.int32())), ("annotated_labels", labels)]
            + [("remarks", text), ("schema_version", text), ("subjects", pa.int32()), ("scans", pa.int32())]
            + [("scans_provided", pa.int32())]
        ),
        "scans": pa.schema(
            [("scan", text)]
            + ids
            + [("source_subject_id", text), ("gender", text), ("source_scan_id", text)]
            + [("providers", pa.list_(pa.int32())), ("image_provider", pa.int32()), ("modality", text)]
            + [("age", pa.int32()), ("weight", pa.float64()), ("height", pa.float64()), ("bmi", pa.float64())]
            + [("postmortem", pa.bool_()), ("days_from_baseline", pa.int32()), ("registered_to", text)]
            + [("source_split", text), ("image", pa.bool_()), ("image_license", text), ("image_published", pa.bool_())]
            + [("image_size", pa.list_(pa.int32())), ("image_spacing", pa.list_(pa.float64()))]
            + [(kind, labels) for kind in LABEL_KINDS]
            + [("truncated", labels)]
            + [("subject_remarks", text), ("remarks", text)]
        ),
        "labels": pa.schema(
            [("scan", text)]
            + ids
            + [("provider", pa.int32()), ("subset", text), ("label", text), ("kind", text), ("origin", text)]
            + [("license", text), ("published", pa.bool_()), ("truncated", pa.bool_())]
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
