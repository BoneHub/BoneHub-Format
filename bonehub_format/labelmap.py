"""The label map: the names of the structures segmentations, meshes and NURBS surfaces hold, and how they relate.

Every label is a structure of a tree (FEMUR, KIDNEY, LEFT_ATRIUM), or a part, tissue or side of one. Its name says
which, always in this order:

    NODE[_PART][_TISSUE][_SIDE]

    FEMUR_LEFT                      the left femur
    FEMUR_PROXIMAL_CORTICAL_LEFT    the cortical bone of the proximal third of the left femur
    VERTEBRA_L3_BODY                the body of the third lumbar vertebra
    KIDNEY_CYST_LEFT                a cyst of the left kidney (findings and devices are 'tissues' of their structure)
    LUNG_UPPER_LOBE_RIGHT           a child structure (the lobe) of LUNG, on the right
    SKULL_WITHOUT_MANDIBLE          an aggregate: a union of labels that is no node of the tree

A name without a side is the structure as a whole: both sides together, or a structure that is not paired. A label
always means the whole of what its name says (a kidney with its cysts and tumors).

The tree has one root per kind of structure: SKELETON, CARTILAGE, LIGAMENTS, JOINTS, MUSCLES, HEART_AND_PERICARDIUM,
ARTERIES, VEINS, LYMPHOID_ORGANS, BRAIN, SPINAL_CORD, CEREBROSPINAL_FLUID, NERVES, EYE_AND_ADNEXA, EAR,
RESPIRATORY_SYSTEM, DIGESTIVE_SYSTEM, URINARY_SYSTEM, GENITAL_ORGANS, ENDOCRINE_GLANDS, INTEGUMENT, ADIPOSE_TISSUE,
BODY_CAVITIES and BODY (body regions). labelmap_spec.py holds it: read it to see where a structure sits.

Landmarks are labels too, of the LANDMARKS root, and only landmarks files hold them. Their names start with LANDMARK_,
then the structure they sit on, then what they are:

    LANDMARK_FEMUR_HEAD_CENTER_LEFT          the centre of the left femoral head (a point, or a fitted sphere)
    LANDMARK_GLUTEUS_MEDIUS_ORIGIN_RIGHT     the origin of the right gluteus medius (points, or an area)
    LANDMARK_VERTEBRA_L3_CENTROID            the centroid of L3

    landmark_on(LabelMap.LANDMARK_FEMUR_HEAD_CENTER_LEFT)     LabelMap.FEMUR_LEFT
    geometry_of(LabelMap.LANDMARK_FEMUR_HEAD_CENTER_LEFT)     ('point', 'sphere')

What a label covers: the labels under it.

    covers(LabelMap.RIBS, LabelMap.RIB_7_LEFT)                  True   (a group covers its members)
    covers(LabelMap.PATELLA, LabelMap.PATELLA_LEFT)             True   (no side covers both sides)
    covers(LabelMap.KIDNEY_LEFT, LabelMap.KIDNEY_CYST_LEFT)     True   (a structure covers what lies in it)
    covers(LabelMap.FEMUR_PROXIMAL, LabelMap.FEMUR_HEAD_LEFT)   True   (the femoral head lies in the proximal femur)
    covers(LabelMap.SKULL, LabelMap.HYOID)                      False  (the hyoid is not part of the skull)
    labels_covered(LabelMap.SKULL_WITHOUT_MANDIBLE)             every label of the cranium and of the facial bones but
                                                                the mandible

The label map only grows: once a database is published, a label keeps its name and meaning, and what it covers among
published labels never changes. Labels are added in labelmap_spec.py; tools/generate_labelmap.py then writes the
LabelMap enum (labelmap_data.py) and checks every rule (tests/test_labelmap.py lists them).
"""

from __future__ import annotations

from enum import Enum

from . import labelmap_build as _build
from .labelmap_data import LabelMap

__all__ = [
    "LabelMap",
    "node_of",
    "part_of",
    "tissue_of",
    "side_of",
    "kind_of",
    "category_of",
    "section_of",
    "parent_of",
    "is_aggregate",
    "members_of",
    "covers",
    "labels_covered",
    "label_for",
    "is_landmark",
    "landmark_on",
    "geometry_of",
]

#: The label map's labels and nodes, built from labelmap_spec.py (see labelmap_build.py).
TABLES = _build.build()


def _name(label: LabelMap | str) -> str:
    name = label.name if isinstance(label, Enum) else str(label)
    if name not in TABLES.labels:
        raise ValueError(f"'{name}' is not a label of the label map (see bonehub_format/labelmap.py).")
    return name


def node_of(label: LabelMap | str) -> str:
    """The structure (node) a label names: 'FEMUR' for FEMUR_PROXIMAL_CORTICAL_LEFT."""
    return TABLES.labels[_name(label)].node


def part_of(label: LabelMap | str) -> str | None:
    """The part a label names ('PROXIMAL'), None for the whole structure."""
    return TABLES.labels[_name(label)].part


def tissue_of(label: LabelMap | str) -> str | None:
    """The tissue, finding or device a label names ('CORTICAL', 'CYST', 'IMPLANT'), None for the whole structure."""
    return TABLES.labels[_name(label)].tissue


def side_of(label: LabelMap | str) -> str | None:
    """'LEFT', 'RIGHT', 'SIDE_UNKNOWN' (one side, which is unknown), or None: both sides, or not paired."""
    return TABLES.labels[_name(label)].side


def kind_of(label: LabelMap | str) -> str:
    """The kind of structure: 'bone', 'cartilage', 'ligament', 'joint', 'muscle', 'heart', 'artery', 'vein', ..."""
    return TABLES.kind_of(_name(label))


def category_of(label: LabelMap | str) -> str:
    """'structure' (a structure or a part of one), 'tissue' (e.g. cortical bone), 'finding' or 'device'."""
    return TABLES.category_of(_name(label))


def section_of(label: LabelMap | str) -> str:
    """The section of the database's webpage a label is listed in ('Leg', 'Arteries', 'Findings', ...)."""
    return TABLES.section_of(_name(label))


def parent_of(node: str) -> str | None:
    """The parent of a structure node in the tree: 'LEG_BONES' for 'FEMUR'; None for a root."""
    return TABLES.nodes[node].parent


def is_aggregate(label: LabelMap | str) -> bool:
    """Whether the label is a union of labels that is no node of the tree (e.g. SKULL_WITHOUT_MANDIBLE)."""
    return TABLES.is_aggregate_node(node_of(label))


def members_of(label: LabelMap | str) -> list[LabelMap]:
    """The labels an aggregate unites (empty for any other label)."""
    return [LabelMap[m] for m in TABLES.nodes[node_of(label)].spec.members or ()]


def covers(a: LabelMap | str, b: LabelMap | str) -> bool:
    """Whether label a covers label b: b is a, or lies under it (see the module docstring)."""
    return TABLES.covers(_name(a), _name(b))


def labels_covered(label: LabelMap | str) -> list[LabelMap]:
    """Every label that `label` covers, itself included, in label map order."""
    a = _name(label)
    return [other for other in LabelMap if TABLES.covers(a, other.name)]


def label_for(node: str, part: str | None = None, tissue: str | None = None, side: str | None = None) -> LabelMap:
    """The label of a structure, part, tissue and side: label_for('FEMUR', 'PROXIMAL', side='LEFT')."""
    return LabelMap[_build._name(node, part, tissue, side)]


def is_landmark(label: LabelMap | str) -> bool:
    """Whether the label is a landmark (LANDMARK_...), which landmarks files hold, not segmentations or models."""
    return TABLES.is_landmark(_name(label))


def landmark_on(label: LabelMap | str) -> LabelMap:
    """The structure a landmark sits on, with the landmark's side: FEMUR_LEFT for LANDMARK_FEMUR_HEAD_CENTER_LEFT."""
    name = _name(label)
    if not TABLES.is_landmark(name):
        raise ValueError(f"{name} is not a landmark.")
    return LabelMap[TABLES.landmark_on(name)]


def geometry_of(label: LabelMap | str) -> tuple[str, ...]:
    """The geometries an entry of a landmark may take: ('point', 'sphere') for LANDMARK_FEMUR_HEAD_CENTER."""
    name = _name(label)
    if not TABLES.is_landmark(name):
        raise ValueError(f"{name} is not a landmark.")
    return TABLES.geometry_of(name)
