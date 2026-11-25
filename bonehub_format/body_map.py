"""Which structures of the body each label covers, grouped the way the database's webpage shows them.

The webpage counts scans per GROUP: the shapes of its skeleton figure (`map` groups, one per side), the rows of its
structure x dataset matrix (`row` groups, sides merged), and the soft-tissue groups (muscles, cartilage, ligaments,
body composition), which are rows too. A label counts for a group when its structure is one of the group's: the
structure itself or a part of it (FEMUR_PROXIMAL_LEFT counts for the left femur), or one of the structures the group
gathers (SCAPHOID_LEFT and CARPALS_LEFT both count for the left carpals). A label of a larger group does not count
for the structures inside it, because its mask does not separate them: a mask of the whole presacral spine is not a
segmentation of each vertebra. Such labels have rows of their own ('... (one mask)').

    structures_data()   the groups and, for every label, the groups it covers; written to structures.json
"""

from __future__ import annotations

from functools import cache

from .labelmap import LabelMap, side_of, structure_of

SIDES = {1: "L", 2: "R"}
SIDE_NAMES = {1: "left", 2: "right"}


@cache
def _all_structures() -> frozenset[int]:
    return frozenset(structure_of(label.value) for label in LabelMap if label.value)


def _where(predicate) -> frozenset[int]:
    return frozenset(s for s in _all_structures() if predicate(s))


def _families(*families: int) -> frozenset[int]:
    """Every structure of these RS families, e.g. 21 for the cervical spine (CERVICAL_SPINE and C1 .. C7)."""
    return _where(lambda s: s // 100 in families)


def _group(gid, name, section, kind, members, side=0, map=False, row=False) -> dict:
    return {"id": gid, "name": name, "section": section, "kind": kind, "side": side, "members": frozenset(members),
            "map": map, "row": row}


def _sided(gid, name, section, members, map=True) -> list[dict]:
    return [_group(f"{gid}-{SIDES[side]}", f"{name} ({SIDE_NAMES[side]})", section, "bone", members, side, map=map) for side in (1, 2)]


@cache
def groups() -> list[dict]:
    """The groups of the webpage, in display order."""
    f = _families
    found = []
    # The skeleton figure: one shape per group, sided where the bone is paired.
    found.append(_group("skull", "Skull (cranium and face)", "Head", "bone", (f(11, 12) - {1201}) | {1000, 1001}, map=True))
    found.append(_group("mandible", "Mandible", "Head", "bone", {1201}, map=True))
    found += [_group(f"C{i}", f"Vertebra C{i}", "Spine", "bone", {2100 + i}, map=True) for i in range(1, 8)]
    found += [_group(f"T{i}", f"Vertebra T{i}", "Spine", "bone", {2200 + i}, map=True) for i in range(1, 13)]
    found += [_group(f"L{i}", f"Vertebra L{i}", "Spine", "bone", {2300 + i}, map=True) for i in range(1, 6)]
    found.append(_group("sacrum", "Sacrum", "Spine", "bone", f(24), map=True))
    found.append(_group("coccyx", "Coccyx", "Spine", "bone", {2500}, map=True))
    for i in range(1, 13):
        found += _sided(f"rib{i}", f"Rib {i}", "Thorax", {3100 + i})
    found.append(_group("sternum", "Sternum", "Thorax", "bone", {3200}, map=True))
    for gid, name, members in (
        ("clavicle", "Clavicle", {4100}),
        ("scapula", "Scapula", {4200}),
        ("humerus", "Humerus", {4300}),
        ("ulna", "Ulna", {4400}),
        ("radius", "Radius", {4500}),
    ):
        found += _sided(gid, name, "Shoulder and arm", members)
    for gid, name, members in (("carpals", "Carpals", f(51)), ("metacarpals", "Metacarpals", f(52)), ("fingers", "Phalanges of the hand", f(53, 54))):
        found += _sided(gid, name, "Hand", members)
    found += _sided("hip", "Hip bone", "Pelvis", {6100})
    for gid, name, members in (("femur", "Femur", {7100}), ("patella", "Patella", {7200}), ("tibia", "Tibia", {7300}), ("fibula", "Fibula", {7400})):
        found += _sided(gid, name, "Leg", members)
    for gid, name, members in (("tarsals", "Tarsals", f(81)), ("metatarsals", "Metatarsals", f(82)), ("toes", "Phalanges of the foot", f(83, 84))):
        found += _sided(gid, name, "Foot", members)

    # The rows of the matrix for the bones: sides merged, the spine and ribs by section.
    rows = [
        ("r-skull", "Skull (cranium and face)", "Head", (f(11, 12) - {1201}) | {1000, 1001}),
        ("r-mandible", "Mandible", "Head", {1201}),
        ("r-hyoid", "Hyoid and auditory ossicles", "Head", f(13, 14)),
        ("r-cervical", "Cervical vertebrae", "Spine", f(21)),
        ("r-thoracic", "Thoracic vertebrae", "Spine", f(22)),
        ("r-lumbar", "Lumbar vertebrae", "Spine", f(23)),
        ("r-sacrum", "Sacrum and coccyx", "Spine", f(24, 25)),
        ("r-ribs", "Ribs", "Thorax", f(31)),
        ("r-sternum", "Sternum", "Thorax", {3200}),
        ("r-clavicle", "Clavicle", "Shoulder and arm", {4100}),
        ("r-scapula", "Scapula", "Shoulder and arm", {4200}),
        ("r-humerus", "Humerus", "Shoulder and arm", {4300}),
        ("r-forearm", "Radius and ulna", "Shoulder and arm", {4400, 4500}),
        ("r-hand", "Hand bones", "Hand", f(51, 52, 53, 54) | {5000}),
        ("r-hip", "Hip bone", "Pelvis", {6100}),
        ("r-femur", "Femur", "Leg", {7100}),
        ("r-patella", "Patella", "Leg", {7200, 7500}),
        ("r-leg", "Tibia and fibula", "Leg", {7300, 7400}),
        ("r-foot", "Foot bones", "Foot", f(81, 82, 83, 84) | {8000}),
        ("r-one-spine", "Vertebral column (one mask)", "Bones in one mask", {2000, 2001, 2002}),
        ("r-one-thorax", "Thoracic cage (one mask)", "Bones in one mask", {3000}),
        ("r-one-arm", "Upper extremity (one mask)", "Bones in one mask", {4000, 4001}),
        ("r-one-pelvis", "Pelvis (one mask)", "Bones in one mask", {6000}),
        ("r-one-leg", "Lower extremity (one mask)", "Bones in one mask", {7000}),
        ("r-one-skeleton", "Skeleton (one mask)", "Bones in one mask", {9000, 9001, 9002, 9900}),
    ]
    found += [_group(gid, name, section, "bone", members, row=True) for gid, name, section, members in rows]

    # Soft tissue: both a list on the page and rows of the matrix.
    soft = [
        ("m-head", "Extraocular muscles", "Muscles", "muscle", f(19)),
        ("m-back", "Deep back muscles", "Muscles", "muscle", f(29)),
        ("m-trunk", "Abdominal wall muscles", "Muscles", "muscle", f(39)),
        ("m-gluteal", "Gluteal muscles", "Muscles", "muscle", _where(lambda s: 6910 <= s <= 6919)),
        ("m-rotators", "Deep hip rotators", "Muscles", "muscle", _where(lambda s: 6920 <= s <= 6929)),
        ("m-iliopsoas", "Iliopsoas", "Muscles", "muscle", _where(lambda s: 6930 <= s <= 6939)),
        ("m-quadriceps", "Quadriceps", "Muscles", "muscle", _where(lambda s: 7910 <= s <= 7919)),
        ("m-adductors", "Thigh adductors", "Muscles", "muscle", _where(lambda s: 7920 <= s <= 7929)),
        ("m-hamstrings", "Hamstrings", "Muscles", "muscle", _where(lambda s: 7930 <= s <= 7939)),
        ("m-sartorius", "Sartorius", "Muscles", "muscle", {7940}),
        ("m-leg-anterior", "Anterior leg muscles", "Muscles", "muscle", _where(lambda s: 7950 <= s <= 7959)),
        ("m-leg-fibular", "Fibular muscles", "Muscles", "muscle", _where(lambda s: 7960 <= s <= 7969)),
        ("m-calf", "Calf muscles", "Muscles", "muscle", _where(lambda s: 7970 <= s <= 7979)),
        ("m-leg-deep", "Deep posterior leg muscles", "Muscles", "muscle", _where(lambda s: 7980 <= s <= 7989)),
        ("m-hip-all", "Hip muscles (one mask)", "Muscles", "muscle", {6900}),
        ("m-leg-all", "Leg muscles (one mask)", "Muscles", "muscle", {7900, 7901, 7902}),
        ("c-discs", "Intervertebral discs", "Cartilage", "cartilage", f(27)),
        ("c-costal", "Costal cartilages", "Cartilage", "cartilage", f(37)),
        ("c-hip", "Hip joint cartilage", "Cartilage", "cartilage", {6700, 6701, 7701}),
        ("c-knee", "Knee cartilage", "Cartilage", "cartilage", {7702, 7703, 7704, 7705}),
        ("c-ankle", "Ankle cartilage", "Cartilage", "cartilage", {7706, 8700, 8701}),
        ("c-leg-all", "Leg cartilage (one mask)", "Cartilage", "cartilage", {7700}),
        ("l-knee", "Knee ligaments", "Ligaments", "ligament", f(78)),
        ("t-muscle", "Skeletal muscle tissue", "Body composition", "tissue", {9100}),
        ("t-fat-sub", "Subcutaneous fat", "Body composition", "tissue", {9201}),
        ("t-fat-visceral", "Visceral fat", "Body composition", "tissue", {9202}),
        ("t-fat-inter", "Intermuscular fat", "Body composition", "tissue", {9203}),
        ("t-fat", "Fat (not divided)", "Body composition", "tissue", {9200}),
        ("t-skin", "Skin", "Body composition", "tissue", {9300}),
    ]
    found += [_group(gid, name, section, kind, members, map=False, row=True) for gid, name, section, kind, members in soft]
    return found


def label_groups(label: LabelMap) -> list[str]:
    """The ids of the groups a label covers: those its structure is one of, on its side (a label whose side is not
    distinguished covers both)."""
    structure, side = structure_of(label.value), side_of(label.value)
    return [
        group["id"]
        for group in groups()
        if structure in group["members"] and (group["side"] == 0 or side == 0 or side == group["side"])
    ]


def structures_data() -> dict:
    """What the webpage needs to count scans per group: the groups (without their members) and, for every label,
    the ids of the groups it covers."""
    return {
        "groups": [{key: value for key, value in group.items() if key != "members"} for group in groups()],
        "labels": {label.name: covered for label in LabelMap if label.value and (covered := label_groups(label))},
    }
