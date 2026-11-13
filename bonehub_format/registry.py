"""The BoneHub registry: the list of source datasets that dataset IDs come from.

Every dataset folder is named after the dataset's BoneHub ID in the registry, the JSON file kept in the BoneHub
Public-Datasets repository: TotalSegmentator CT, BoneHub ID 27, is ds-0027 in every database that holds it. A dataset
made from another one and published as a dataset of its own (see DatasetInfo.derived_from) gets an entry of its own in
the registry first.

The registry is a JSON array with one object per dataset; this package reads the "BoneHub ID" (a positive integer)
and "Dataset Name" fields of each.

    registry = load_registry()                        # BoneHub ID -> dataset name, from REGISTRY_URL
    registry = load_registry("path/to/registry.json")  # or from a local copy
    registry_entry_url(27)                            # the page of TotalSegmentator CT in the Dataset Browser

Needs only the standard library.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from pathlib import Path

#: The registry JSON, as read by load_registry, and the page where people browse it (the Dataset Browser).
REGISTRY_URL = "https://raw.githubusercontent.com/BoneHub/Public-Datasets/main/data/bonehub_public_datasets.json"
REGISTRY_PAGE_URL = "https://bonehub.github.io/Public-Datasets/"
ID_FIELD = "BoneHub ID"
NAME_FIELD = "Dataset Name"


def registry_entry_url(dataset_id: int) -> str:
    """The page of a dataset in the Dataset Browser, which opens its registry entry."""
    return f"{REGISTRY_PAGE_URL}#dataset={dataset_id}"


def parse_registry(text: str) -> dict[int, str]:
    """BoneHub ID -> dataset name, from the text of the registry JSON."""
    try:
        entries = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"The registry is not valid JSON: {e}") from e
    if not isinstance(entries, list):
        raise ValueError("The registry is not a JSON array of datasets.")
    registry: dict[int, str] = {}
    for number, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {number} of the registry is not a JSON object.")
        dataset_id, name = entry.get(ID_FIELD), entry.get(NAME_FIELD)
        if isinstance(dataset_id, bool) or not isinstance(dataset_id, int) or dataset_id < 1:
            raise ValueError(f"Entry {number} of the registry: {ID_FIELD} {dataset_id!r} is not a BoneHub ID.")
        if not isinstance(name, str):
            raise ValueError(f"Entry {number} of the registry (BoneHub ID {dataset_id}) has no {NAME_FIELD}.")
        if dataset_id in registry:
            raise ValueError(f"Entry {number} of the registry: BoneHub ID {dataset_id} is listed twice.")
        registry[dataset_id] = name.strip()
    return registry


def load_registry(source: str | Path = REGISTRY_URL, timeout: float = 30) -> dict[int, str]:
    """BoneHub ID -> dataset name, read from the registry JSON at a URL (http or https) or a local path."""
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=timeout) as response:
            text = response.read().decode("utf-8-sig")
    else:
        text = Path(source).read_text(encoding="utf-8-sig")
    return parse_registry(text)


def as_registry(registry: Mapping[int, str] | str | Path) -> Mapping[int, str]:
    """A registry given as a mapping, or loaded from a URL or path."""
    return registry if isinstance(registry, Mapping) else load_registry(registry)
