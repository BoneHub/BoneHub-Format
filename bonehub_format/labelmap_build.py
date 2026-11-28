"""Building the label map from labelmap_spec.py: its labels and nodes, the queries over them, the checks of its
invariants, and the source of labelmap_data.py (the LabelMap enum).

    tables = build()            # from the spec (or a changed copy of it, in the tests)
    check(tables)               # every invariant; [] when the label map is sound
    render(tables)              # the source of labelmap_data.py

labelmap.py builds the tables once, at import, and answers its queries from them. tools/generate_labelmap.py writes
labelmap_data.py; tests/test_labelmap.py fails when that file is not what the spec gives.
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import labelmap_spec as default_spec

MAX_NODE_NAME, MAX_NAME = 40, 64
#: A landmark's name holds its structure's: its node may be as long as a label leaves room for, with its side.
MAX_LANDMARK_NODE_NAME = MAX_NAME - len("_RIGHT")
#: The muscles, bones and their joints: kinds whose groups are paired when all their children are.
MUSCULOSKELETAL = ("SKELETON", "CARTILAGE", "LIGAMENTS", "JOINTS", "MUSCLES")


@dataclass(frozen=True)
class Label:
    """A label: a structure node, and the part, tissue and side of it the label names (None where it has none)."""

    name: str
    node: str
    part: str | None = None
    tissue: str | None = None
    side: str | None = None

    @property
    def key(self) -> tuple:
        return (self.node, self.part, self.tissue, self.side)


@dataclass
class Node:
    """A node of the tree, with what the spec declares for it."""

    name: str
    parent: str | None
    kind: str  # the name of its tree's root
    policy: str  # paired, halves or single
    only: str | None
    spec: object  # the spec's N
    children: list[str] = field(default_factory=list)


class Tables:
    """The label map: its labels and nodes, and the queries over them."""

    def __init__(self, spec, labels: dict[str, Label], nodes: dict[str, Node], problems: list[str]):
        self.spec, self.labels, self.nodes, self.problems = spec, labels, nodes, problems
        self.tissues = spec.TISSUES
        self.part_broader = {}  # node -> part -> broader parts: the declarations of its line, the nearest winning
        for name in nodes:
            merged = {}
            for ancestor in reversed(self.tree_line(name)):
                merged.update(nodes[ancestor].spec.part_broader)
            self.part_broader[name] = {p: tuple(q) for p, q in merged.items() if q}
        self._memo: dict[tuple, bool] = {}
        self._ancestors: dict[str, frozenset[str]] = {}

    # ----------------------------------------------------------------------------------------- the tree
    def tree_line(self, node: str) -> list[str]:
        """The node and its tree ancestors, up to the root."""
        line = []
        while node is not None:
            line.append(node)
            node = self.nodes[node].parent
        return line

    def ancestors(self, node: str) -> frozenset[str]:
        """The node, its tree ancestors and its extra parents (also_under), theirs too."""
        if node not in self._ancestors:
            out, todo = set(), [node]
            while todo:
                n = todo.pop()
                if n in out or n not in self.nodes:
                    continue
                out.add(n)
                if self.nodes[n].parent:
                    todo.append(self.nodes[n].parent)
                todo.extend(self.nodes[n].spec.also_under)
            self._ancestors[node] = frozenset(out)
        return self._ancestors[node]

    def is_aggregate_node(self, node: str) -> bool:
        return self.nodes[node].spec.members is not None

    # ----------------------------------------------------------------------------------------- what a label is
    def kind_of(self, name: str) -> str:
        return self.spec.KINDS[self.nodes[self.labels[name].node].kind][0]

    def is_landmark(self, name: str) -> bool:
        return self.nodes[self.labels[name].node].kind == "LANDMARKS"

    def landmark_on(self, name: str) -> str:
        """The label of the structure a landmark sits on, with the landmark's side where the structure has one."""
        label = self.labels[name]
        on = self.nodes[label.node].spec.on
        sided = f"{on}_{label.side}" if label.side else on
        return sided if sided in self.labels else on

    def geometry_of(self, name: str) -> tuple[str, ...]:
        return self.nodes[self.labels[name].node].spec.geometry

    def category_of(self, name: str) -> str:
        tissue = self.labels[name].tissue
        return self.tissues[tissue][0] if tissue else "structure"

    def section_of(self, name: str) -> str:
        label, category = self.labels[name], self.category_of(name)
        if category == "finding":
            return "Findings"
        if category == "device":
            return "Devices"
        node = self.nodes[label.node]
        if node.kind == "SKELETON":
            line = self.tree_line(label.node)
            region = line[-2] if len(line) >= 2 else None
            return self.spec.BONE_SECTIONS.get(region, "Whole skeleton")
        return self.spec.KINDS[node.kind][1]

    # ----------------------------------------------------------------------------------------- covers
    def _part_within(self, node: str, outer: str, inner: str) -> bool:
        if outer == inner:
            return True
        broader, todo, seen = self.part_broader.get(node, {}), [inner], set()
        while todo:
            p = todo.pop()
            if p in seen:
                continue
            seen.add(p)
            for q in broader.get(p, ()):
                if q == outer:
                    return True
                todo.append(q)
        return False

    def _tissue_within(self, outer: str | None, inner: str | None) -> bool:
        if outer is None or outer == inner:
            return True
        if inner is None:
            return False
        todo, seen = list(self.tissues[inner][1]), set()
        while todo:
            t = todo.pop()
            if t in seen:
                continue
            seen.add(t)
            if t == outer:
                return True
            todo.extend(self.tissues[t][1])
        return False

    def _part_ok(self, a: tuple, b: tuple) -> bool:
        a_node, a_part, b_node, b_part = a[0], a[1], b[0], b[1]
        if a_part is None:
            return True
        if a_node == b_node:
            return b_part is not None and self._part_within(b_node, a_part, b_part)
        line = self.tree_line(b_node)
        if a_node not in line:  # a part covers only what lies under its node in the tree, not via extra parents
            return False
        if b_part is not None and self._part_within(b_node, a_part, b_part):
            return True
        child = self.nodes[line[line.index(a_node) - 1]]  # the node just below a's on b's line
        return child.spec.in_part is not None and self._part_within(a_node, a_part, child.spec.in_part)

    def _with(self, member: str, of: tuple) -> tuple:
        """A member of an aggregate, with the aggregate label's tissue and side where it has one."""
        m = self.labels[member]
        return (m.node, m.part, of[2] or m.tissue, of[3] or m.side)

    def covers_key(self, a: tuple, b: tuple) -> bool:
        """Whether (node, part, tissue, side) a covers b; see labelmap.covers."""
        if a == b:
            return True
        memo = self._memo.get((a, b))
        if memo is not None:
            return memo
        self._memo[(a, b)] = False  # guards against a cycle while this answer is worked out
        result = False
        if self.is_aggregate_node(a[0]) and a[1] is None:
            result = any(self.covers_key(self._with(m, a), b) for m in self.nodes[a[0]].spec.members)
        if not result:
            if self.is_aggregate_node(b[0]) and b[1] is None and a[0] not in self.ancestors(b[0]):
                result = all(self.covers_key(a, self._with(m, b)) for m in self.nodes[b[0]].spec.members)
            elif a[0] in self.ancestors(b[0]):
                result = self._part_ok(a, b) and self._tissue_within(a[2], b[2]) and a[3] in (None, b[3])
        self._memo[(a, b)] = result
        return result

    def covers(self, a: str, b: str) -> bool:
        return self.covers_key(self.labels[a].key, self.labels[b].key)


def build(spec=default_spec) -> Tables:
    """Every label and node of the spec, in label map order. The spec's objects are copied, so a test may pass a
    changed copy."""
    tree, roots = copy.deepcopy(spec.TREE), copy.deepcopy(spec.ROOTS)
    labels: dict[str, Label] = {}
    nodes: dict[str, Node] = {}
    problems: list[str] = []

    def add(label: Label):
        if label.name in labels:
            problems.append(f"duplicate label name {label.name}")
        else:
            labels[label.name] = label

    def emit(node: Node):
        s = node.spec
        sides = [node.only] if node.only else ([None, *spec.SIDES] if node.policy in ("paired", "halves") else [None])
        for side in sides:
            add(Label(_name(s.name, None, None, side), s.name, side=side))
        for part in s.parts:
            for side in sides:
                add(Label(_name(s.name, part, None, side), s.name, part, None, side))
            if part in s.part_tissues:
                for tissue in s.tissues:
                    if tissue == "MEDULLARY_CAVITY" and part not in ("PROXIMAL", "SHAFT", "DISTAL"):
                        continue
                    for side in sides:
                        add(Label(_name(s.name, part, tissue, side), s.name, part, tissue, side))
        for tissue in s.tissues:
            if tissue not in spec.TISSUES:
                problems.append(f"{s.name}: tissue {tissue} is not in TISSUES")
                continue
            for side in sides:
                add(Label(_name(s.name, None, tissue, side), s.name, None, tissue, side))

    def walk(s, parent: str | None, kind: str, parent_policy: str | None, parent_only: str | None, labelled=True):
        if s.name in nodes:
            problems.append(f"duplicate node name {s.name}")
            return
        policy = s.side or ("paired" if parent_policy == "paired" else "single")
        only = s.only or parent_only
        nodes[s.name] = Node(s.name, parent, kind, policy, only, s)
        if parent:
            nodes[parent].children.append(s.name)
        if labelled:
            emit(nodes[s.name])
        for child in s.children:
            walk(child, s.name, kind, policy, only)

    for root in spec.KINDS:
        s = roots.get(root) or spec.N(root)
        s.children = tree.get(root, [])
        # GROUPS_ACROSS_KINDS holds the aggregates of several kinds, LANDMARKS the landmarks; neither is a label itself
        walk(s, None, root, None, None, labelled=root not in spec.UNLABELLED_ROOTS)
    return Tables(spec, labels, nodes, problems)


def _name(node: str, part: str | None, tissue: str | None, side: str | None) -> str:
    return "_".join(x for x in (node, part, tissue, side) if x)


# --------------------------------------------------------------------------------------------- invariants
def _acyclic(graph: dict) -> bool:
    state = {}

    def visit(u):
        state[u] = 1
        for v in graph.get(u, ()):
            if state.get(v) == 1 or (state.get(v) is None and not visit(v)):
                return False
        state[u] = 2
        return True

    return all(state.get(u) == 2 or visit(u) for u in list(graph))


def check(t: Tables, published: list[str] | None = None, published_covers: dict[str, list[str]] | None = None) -> list[str]:
    """Every invariant of the label map: an empty list when all hold. `published` (the names of earlier releases) and
    `published_covers` (what each covers among them) are what the releases froze."""
    spec, p = t.spec, list(t.problems)
    for name, label in t.labels.items():
        if name != _name(*label.key):
            p.append(f"{name}: its fields spell {_name(*label.key)}")
        if len(name) > MAX_NAME:
            p.append(f"{name}: longer than {MAX_NAME} characters")
        node = t.nodes[label.node]
        less = [(label.node, label.part, label.tissue, None if not node.only else label.side),
                (label.node, label.part, None, label.side), (label.node, None, label.tissue, label.side)]
        for key in less:
            if _name(*key) not in t.labels:
                p.append(f"{name}: the less specific label {_name(*key)} does not exist")
        if label.side == spec.SIDE_UNKNOWN and node.policy != "paired":
            p.append(f"{name}: an unknown side on a structure that is not paired")
    for name, node in t.nodes.items():
        s = node.spec
        if node.kind not in spec.UNLABELLED_ROOTS or node.parent:
            if not any(_name(name, None, None, side) in t.labels for side in (None, *spec.SIDES)):
                p.append(f"node {name} has no label")
        longest = MAX_LANDMARK_NODE_NAME if node.kind == "LANDMARKS" else MAX_NODE_NAME
        if len(name) > longest:
            p.append(f"node {name}: longer than {longest} characters")
        p.extend(_landmark_problems(t, name, node))
        for word in (*spec.SIDES, spec.SIDE_UNKNOWN):
            if name.endswith("_" + word):
                p.append(f"node {name} ends in the side word {word}")
        for cut in (i for i, ch in enumerate(name) if ch == "_"):  # it must not read as another node + its part or tissue
            other, word = name[:cut], name[cut + 1:]
            if other in t.nodes and (word in t.nodes[other].spec.parts or word in t.nodes[other].spec.tissues):
                p.append(f"node {name} reads as node {other} + {word}")
        if s.only and node.policy != "paired":
            p.append(f"{name}: 'only' on a structure that is not paired")
        below = {part for n in t.nodes if name in t.tree_line(n) for part in t.nodes[n].spec.parts}
        for part, broader in s.part_broader.items():
            for x in (part, *broader):
                if x not in below:
                    p.append(f"{name}: the narrower-part declaration names {x}, which no node of its line has")
        if not _acyclic({q: list(b) for q, b in s.part_broader.items()}):
            p.append(f"{name}: narrower parts form a cycle")
        if s.in_part and (not node.parent or s.in_part not in t.nodes[node.parent].spec.parts):
            p.append(f"{name}: in_part {s.in_part} is not a part of its parent")
        for parent in s.also_under:
            if parent not in t.nodes:
                p.append(f"{name}: extra parent {parent} is not a node")
        if s.members is not None and s.children:
            p.append(f"{name}: an aggregate has no children")
    if not _acyclic({n: [x for x in (t.nodes[n].parent, *t.nodes[n].spec.also_under) if x] for n in t.nodes}):
        p.append("the tree with its extra parents has a cycle")
    for tissue, (category, broader) in spec.TISSUES.items():
        for x in broader:
            if x not in spec.TISSUES:
                p.append(f"tissue {tissue}: broader {x} is not a tissue")
            elif spec.TISSUES[x][0] != category and not (category == "finding" and spec.TISSUES[x][0] == "tissue"):
                p.append(f"tissue {tissue} ({category}): broader {x} is a {spec.TISSUES[x][0]}")
    if not _acyclic({x: b for x, (_, b) in spec.TISSUES.items()}):
        p.append("narrower tissues form a cycle")
    aggregates = {}
    for name, node in t.nodes.items():
        members = node.spec.members
        if members is None:
            continue
        missing = [m for m in members if m not in t.labels]
        if missing:
            p.append(f"{name}: members {missing} are not labels")
            continue
        member_nodes = [t.labels[m].node for m in members]
        aggregates[name] = [m for m in member_nodes if t.is_aggregate_node(m)]
        kinds = {t.nodes[m].kind for m in member_nodes}
        if node.kind == "GROUPS_ACROSS_KINDS":
            if len(kinds) < 2:
                p.append(f"{name}: a group across kinds has members of one kind")
        elif kinds != {node.kind}:
            p.append(f"{name}: a member of another kind (a group across kinds belongs in GROUPS_ACROSS_KINDS)")
        else:
            lines = [t.tree_line(m) for m in member_nodes]
            common = [n for n in lines[0] if all(n in line for line in lines)][0]
            if node.parent != common:
                p.append(f"{name}: sits under {node.parent}, but its members' nearest common ancestor is {common}")
        sided = [m for m in members if t.labels[m].side]
        if node.policy == "paired":
            if sided:
                p.append(f"{name}: a paired aggregate with sided members {sided}")
            if not any(t.nodes[m].policy == "paired" for m in member_nodes):
                p.append(f"{name}: a paired aggregate with no paired member")
        elif not sided and all(t.nodes[m].policy == "paired" for m in member_nodes):
            p.append(f"{name}: all its members are paired, so it is paired too")
    if not _acyclic(aggregates):
        p.append("an aggregate contains itself")
    for name, node in t.nodes.items():
        if node.kind in MUSCULOSKELETAL:
            regular = [c for c in node.children if not t.is_aggregate_node(c)]
            if regular and all(t.nodes[c].policy == "paired" for c in regular) and node.policy != "paired":
                p.append(f"{name}: all its children are paired, so it is paired too")
    for name in published or ():
        if name not in t.labels:
            p.append(f"the published label {name} was renamed or removed")
    if published_covers:
        now = covers_snapshot(t, list(published_covers))
        for name, covered in published_covers.items():
            if now.get(name) != covered:
                p.append(f"what the published label {name} covers changed")
    return p


def _landmark_problems(t: Tables, name: str, node: Node) -> list[str]:
    """A landmark node names a structure of another kind and its geometries, and nothing else; no other node does."""
    spec, s, p = t.spec, node.spec, []
    if node.kind != "LANDMARKS":
        if name.startswith(spec.LANDMARK_PREFIX) or s.on is not None or s.geometry:
            p.append(f"node {name}: only a landmark is named {spec.LANDMARK_PREFIX}... and has on= and geometry=")
        return p
    if not node.parent:
        return p
    if not name.startswith(f"{spec.LANDMARK_PREFIX}{s.on}_"):
        p.append(f"landmark {name}: its name does not start with {spec.LANDMARK_PREFIX}{s.on}_")
    if s.on not in t.labels or t.nodes[t.labels[s.on].node].kind == "LANDMARKS":
        p.append(f"landmark {name}: on={s.on} is not a structure label")
    elif node.policy == "paired" and f"{s.on}_LEFT" not in t.labels:
        p.append(f"landmark {name}: paired, but {s.on} has no sides")
    if not s.geometry or any(g not in spec.GEOMETRIES for g in s.geometry):
        p.append(f"landmark {name}: geometry {s.geometry} is not among {spec.GEOMETRIES}")
    if s.children or s.parts or s.tissues or s.members is not None:
        p.append(f"landmark {name}: a landmark has no children, parts, tissues or members")
    return p


def covers_snapshot(t: Tables, names: list[str]) -> dict[str, list[str]]:
    """For each of `names`, the labels among `names` it covers: what a release freezes."""
    names = [n for n in names if n in t.labels]
    return {a: [b for b in names if t.covers(a, b)] for a in names}


def read_published(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """The published names (labelmap_published.csv) and what each covers (labelmap_published_covers.json)."""
    path = Path(path)
    if not path.is_file():
        return [], {}
    with open(path, encoding="utf-8") as f:
        names = [row["name"] for row in csv.DictReader(f)]
    snapshot = path.with_name("labelmap_published_covers.json")
    return names, json.loads(snapshot.read_text(encoding="utf-8")) if snapshot.is_file() else {}


# --------------------------------------------------------------------------------------------- labelmap_data.py
def render(t: Tables) -> str:
    """The source of labelmap_data.py: the LabelMap enum, every label in label map order, by kind."""
    out = [
        '"""Every label of the label map. GENERATED from labelmap_spec.py by tools/generate_labelmap.py: do not edit.',
        "",
        "A label's value is its name. What a label is (its structure, part, tissue and side), what it belongs to and",
        'what it covers: see labelmap.py."""',
        "",
        "from enum import Enum, unique",
        "",
        "",
        "@unique",
        "class LabelMap(Enum):",
        '    """Every label of the label map; see labelmap.py."""',
    ]
    kind = None
    for name, label in t.labels.items():
        node = t.nodes[label.node]
        if node.kind != kind:
            kind = node.kind
            out += ["", f"    # {t.spec.KINDS[kind][0]}"]
        note = node.spec.note if name == node.name and node.spec.note else ""
        out.append(f'    {name} = "{name}"' + (f"  # {note}" if note else ""))
    out += ["", "    @classmethod", "    def get_names_list(cls):", "        return [label.name for label in cls]", ""]
    return "\n".join(out)
