"""Matches between scans, kept in links.json at the root.

    {"links": [{"scans": ["ds-002/sub-000012/scan-01", "ds-007/sub-000300/scan-01"],
                "relation": "same_image", "evidence": "identical_slices", "score": 87,
                "decision": "confirmed", "note": null}]}

same_image    the two scans are the same acquisition, possibly re-saved, reoriented, cropped or resampled
same_subject  the two scans show the same person

A same_image link implies same_subject. Only confirmed links count; a rejected link is kept so that a matcher does
not propose it again. The links of a derived dataset to the dataset it was made from (see DatasetInfo.derived_from)
are declared by its metadata and are not stored here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from . import layout
from .metadata import write_json

Relation = Literal["same_image", "same_subject"]
Decision = Literal["confirmed", "rejected", "pending"]


class Link(BaseModel):
    scans: tuple[str, str]
    relation: Relation
    evidence: str
    score: float | None = None
    decision: Decision = "pending"
    note: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("scans")
    @classmethod
    def _check_scans(cls, scans):
        for key in scans:
            layout.parse_scan_key(key)
        if scans[0] == scans[1]:
            raise ValueError("A link joins two different scans.")
        return tuple(sorted(scans))

    @property
    def identity(self) -> tuple[str, str, str]:
        return (*self.scans, self.relation)


def load_links(root: Path) -> list[Link]:
    path = Path(root) / layout.LINKS_FILE
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [Link.model_validate(link) for link in json.load(f).get("links", [])]


def save_links(root: Path, links: Iterable[Link]) -> None:
    links = sorted(links, key=lambda link: link.identity)
    write_json(Path(root) / layout.LINKS_FILE, {"links": [link.model_dump(mode="json") for link in links]})


def merge_links(existing: Iterable[Link], found: Iterable[Link]) -> list[Link]:
    """The existing links updated with those found again, and the new ones added.

    A link found again takes the new evidence and score, but keeps the decision already made about it.
    """
    merged = {link.identity: link for link in existing}
    for link in found:
        previous = merged.get(link.identity)
        if previous is None:
            merged[link.identity] = link
        else:
            merged[link.identity] = previous.model_copy(update={"evidence": link.evidence, "score": link.score})
    return list(merged.values())


def declared_links(root) -> list[Link]:
    """same_image links from each scan of a derived dataset to the scan it was made from."""
    from .reader import Root

    root = root if isinstance(root, Root) else Root(root)
    links = []
    for dataset in root.datasets():
        if not dataset.info.derived_from:
            continue
        parents = {layout.dataset_dir_name(dataset_id) for dataset_id in dataset.info.derived_from}
        for scan in dataset.scans():
            source = scan.info.source_scan_id
            if source.split("/", 1)[0] in parents:
                links.append(Link(scans=(scan.key, source), relation="same_image", evidence="declared", decision="confirmed"))
    return links


class _Groups:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, a: str, b: str) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)  # the group is named after its first member


def scan_groups(scan_keys: Iterable[str], links: Iterable[Link]) -> tuple[dict[str, str], dict[str, str]]:
    """Per scan, its image group and its subject group, from the confirmed links between the given scans.

    An image group holds the scans that are the same image and is named after its first scan. A subject group holds
    the scans of one person: those of the same subject folder and those joined by any confirmed link; it is named
    after its first subject folder, e.g. 'ds-002/sub-000012'.
    """
    keys = sorted(set(scan_keys))
    known = set(keys)
    images, subjects = _Groups(), _Groups()
    for key in keys:
        images.find(key)
        subjects.find(key.rsplit("/", 1)[0])
    for link in links:
        if link.decision != "confirmed" or not set(link.scans) <= known:
            continue
        a, b = link.scans
        if link.relation == "same_image":
            images.join(a, b)
        subjects.join(a.rsplit("/", 1)[0], b.rsplit("/", 1)[0])
    image_group = {key: images.find(key) for key in keys}
    subject_group = {key: subjects.find(key.rsplit("/", 1)[0]) for key in keys}
    return image_group, subject_group
