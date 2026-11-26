"""The part of the body each label belongs to, as the database's webpage groups its structures.

A label's section follows from its structure code (labelmap.py, R S M M): the bones of each region (S 0 to 6), and,
across regions, cartilage (S 7), ligaments (S 8) and muscles (S 9). Region 9 holds the whole skeleton and the body
composition classes.

    section_of(LabelMap.FEMUR_LEFT)    'Leg'
    structures_data()                  the sections in display order and the section of every label
"""

from __future__ import annotations

from .labelmap import LabelMap, structure_of

#: The sections in the order the webpage shows them.
SECTIONS: tuple[str, ...] = (
    "Head",
    "Spine",
    "Thorax",
    "Shoulder and arm",
    "Hand",
    "Pelvis",
    "Leg",
    "Foot",
    "Whole skeleton",
    "Muscles",
    "Cartilage",
    "Ligaments",
    "Body composition",
)
_REGIONS = {1: "Head", 2: "Spine", 3: "Thorax", 4: "Shoulder and arm", 5: "Hand", 6: "Pelvis", 7: "Leg", 8: "Foot"}
_ACROSS = {7: "Cartilage", 8: "Ligaments", 9: "Muscles"}


def section_of(label: LabelMap) -> str:
    """The section of the webpage a label belongs to."""
    structure = structure_of(label.value)
    region, kind = structure // 1000, (structure // 100) % 10
    if region == 9:
        return "Body composition" if kind in (1, 2, 3) else "Whole skeleton"
    return _ACROSS.get(kind) or _REGIONS[region]


def structures_data() -> dict:
    """The sections in display order, and the index of the section of every label."""
    return {
        "sections": list(SECTIONS),
        "labels": {label.name: SECTIONS.index(section_of(label)) for label in LabelMap if label.value},
    }
