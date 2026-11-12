"""The BoneHub registry: the list of source datasets that dataset IDs come from.

Every dataset folder is named after the dataset's BoneHub ID in the registry, the table kept in the BoneHub
Public-Datasets repository: TotalSegmentator CT, BoneHub ID 27, is ds-0027 in every database that holds it. A dataset
made from another one and published as a dataset of its own (see DatasetInfo.derived_from) gets a row of its own in
the registry first.

    registry = load_registry()                       # BoneHub ID -> dataset name, from REGISTRY_URL
    registry = load_registry("path/to/registry.csv")  # or from a local copy

Needs only the standard library.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from collections.abc import Mapping
from pathlib import Path

#: The registry CSV, as read by load_registry, and its page for people.
REGISTRY_URL = "https://raw.githubusercontent.com/BoneHub/Public-Datasets/main/data/bonehub_public_datasets.csv"
REGISTRY_PAGE_URL = "https://github.com/BoneHub/Public-Datasets/blob/main/data/bonehub_public_datasets.csv"
ID_COLUMN = "BoneHub ID"
NAME_COLUMN = "Dataset Name"


def parse_registry(text: str) -> dict[int, str]:
    """BoneHub ID -> dataset name, from the text of the registry CSV."""
    reader = csv.DictReader(io.StringIO(text))
    missing = [column for column in (ID_COLUMN, NAME_COLUMN) if column not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"The registry has no {' or '.join(repr(c) for c in missing)} column.")
    registry: dict[int, str] = {}
    for line, row in enumerate(reader, start=2):
        raw_id = (row[ID_COLUMN] or "").strip()
        if not raw_id.isdigit() or int(raw_id) < 1:
            raise ValueError(f"Line {line} of the registry: '{raw_id}' is not a BoneHub ID.")
        dataset_id = int(raw_id)
        if dataset_id in registry:
            raise ValueError(f"Line {line} of the registry: BoneHub ID {dataset_id} is listed twice.")
        registry[dataset_id] = (row[NAME_COLUMN] or "").strip()
    return registry


def load_registry(source: str | Path = REGISTRY_URL, timeout: float = 30) -> dict[int, str]:
    """BoneHub ID -> dataset name, read from the registry CSV at a URL (http or https) or a local path."""
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=timeout) as response:
            text = response.read().decode("utf-8-sig")
    else:
        text = Path(source).read_text(encoding="utf-8-sig")
    return parse_registry(text)


def as_registry(registry: Mapping[int, str] | str | Path) -> Mapping[int, str]:
    """A registry given as a mapping, or loaded from a URL or path."""
    return registry if isinstance(registry, Mapping) else load_registry(registry)
