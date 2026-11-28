"""The sections of the database's webpage, and what the page needs to tell which labels a label covers.

A label's section (labelmap.section_of): bones by body region, every other kind of structure by its kind, and findings
and devices in sections of their own.

    section_of(LabelMap.FEMUR_LEFT)    'Leg'
    structures_data()                  the sections in display order, each label's section, and the label map's tree
                                       for the page's covers() (webpage_template/app.js)
"""

from __future__ import annotations

from . import labelmap
from .labelmap import TABLES, LabelMap

#: The sections in the order the webpage shows them.
SECTIONS: tuple[str, ...] = (
    *TABLES.spec.BONE_SECTIONS.values(),
    "Whole skeleton",
    *dict.fromkeys(section for _, section in TABLES.spec.KINDS.values() if section),
    "Findings",
    "Devices",
)


def section_of(label: LabelMap) -> str:
    """The section of the database's webpage a label belongs to."""
    return labelmap.section_of(label)


def structures_data() -> dict:
    """The sections in display order, the section of every label, and what the page's covers() needs: each label's
    node, part, tissue and side, and each node's parent, extra parents, members (aggregates), the part of its parent it
    lies in and its narrower parts, with the broader tissues of each tissue."""
    return {
        "sections": list(SECTIONS),
        "labels": {label.name: SECTIONS.index(section_of(label)) for label in LabelMap},
        "fields": {name: [l.node, l.part, l.tissue, l.side] for name, l in TABLES.labels.items()},
        "nodes": {
            name: {
                k: v
                for k, v in (
                    ("parent", node.parent),
                    ("alsoUnder", list(node.spec.also_under)),
                    ("members", list(node.spec.members) if node.spec.members is not None else None),
                    ("inPart", node.spec.in_part),
                    ("partBroader", {p: list(q) for p, q in TABLES.part_broader[name].items()}),
                )
                if v
            }
            for name, node in TABLES.nodes.items()
        },
        "tissueBroader": {tissue: list(broader) for tissue, (_, broader) in TABLES.tissues.items() if broader},
    }
