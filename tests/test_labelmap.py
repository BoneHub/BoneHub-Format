"""The label map's rules: what tools/generate_labelmap.py checks, the coverage answers anatomy expects, and the
webpage's JavaScript port of covers() giving the same answers as Python."""

import copy
import json
import random
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

from bonehub_format import LabelMap as LM
from bonehub_format import (
    category_of,
    covers,
    is_aggregate,
    kind_of,
    labelmap_build,
    labelmap_spec,
    labels_covered,
    members_of,
    node_of,
    part_of,
    section_of,
    side_of,
    tissue_of,
)
from bonehub_format.body_map import SECTIONS, structures_data
from bonehub_format.labelmap import TABLES

PACKAGE = Path(labelmap_build.__file__).parent
NODE = shutil.which("node") or (p if (p := Path("/root/miniconda3/envs/bonehub-web/bin/node")).is_file() else None)

#: (a, b, whether a covers b): the answers anatomy gives.
GOLDEN = [
    # sides, groups, the tree
    ("PATELLA", "PATELLA_LEFT", True), ("PATELLA_LEFT", "PATELLA_RIGHT", False), ("RIBS", "RIB_7_LEFT", True),
    ("RIBS_LEFT", "RIB_7_RIGHT", False), ("RIBS", "STERNUM", False), ("PRESACRAL_SPINE", "VERTEBRA_L6", True),
    ("PRESACRAL_SPINE", "SACRUM", False), ("VERTEBRAL_COLUMN", "VERTEBRA_S1_BODY_CORTICAL", True),
    ("SKULL", "HYOID", False), ("SKULL", "MANDIBLE_LEFT", True), ("HEAD_BONES", "HYOID", True),
    ("SKELETON", "FEMUR_PROXIMAL_CORTICAL_LEFT", True), ("SKELETON", "MUSCLES", False),
    ("PHALANGES_HAND_1", "PROXIMAL_PHALANX_HAND_1_LEFT", True), ("PHALANGES_HAND_1", "PROXIMAL_PHALANX_HAND_2", False),
    ("QUADRICEPS_FEMORIS", "RECTUS_FEMORIS_LEFT", True), ("MUSCLES", "GLUTEUS_MAXIMUS_RIGHT", True),
    # aggregates: members, sides, aggregates of aggregates
    ("SKULL_WITHOUT_MANDIBLE", "MAXILLA_LEFT", True), ("SKULL_WITHOUT_MANDIBLE", "MANDIBLE", False),
    ("UPPER_EXTREMITY_LEFT", "DISTAL_PHALANX_HAND_3_LEFT", True), ("UPPER_EXTREMITY_LEFT", "HUMERUS_RIGHT", False),
    ("APPENDICULAR_SKELETON", "UPPER_EXTREMITY_LEFT", True), ("APPENDICULAR_SKELETON", "LOWER_EXTREMITY", True),
    ("BONY_PELVIS", "VERTEBRA_S2", True), ("BONY_PELVIS", "VERTEBRA_L5", False),
    ("SACRUM_S2_S5", "VERTEBRA_S3_BODY", True), ("SACRUM_S2_S5", "VERTEBRA_S1", False),
    ("PORTAL_VEIN_AND_SPLENIC_VEIN", "PORTAL_VEIN", True), ("PORTAL_VEIN_AND_SPLENIC_VEIN", "SUPERIOR_MESENTERIC_VEIN", False),
    ("PROSTATE_OR_UTERUS", "UTERINE_CERVIX", True), ("GONADS_LEFT", "OVARY_LEFT", True), ("GONADS_LEFT", "OVARY_RIGHT", False),
    ("HEART_ATRIA", "LEFT_ATRIAL_APPENDAGE", True), ("HEART_ATRIA", "LEFT_VENTRICLE", False),
    ("HEAD_VESSELS_LEFT", "MIDDLE_CEREBRAL_ARTERY_M1_LEFT", True), ("HEAD_VESSELS_LEFT", "INTERNAL_CAROTID_ARTERY_C6_LEFT", True),
    ("HEAD_VESSELS", "INTERNAL_CAROTID_ARTERY_C2_LEFT", False), ("HEAD_VESSELS", "AORTA", False),
    ("HEAD_AND_NECK_GLANDS", "PAROTID_GLAND_LEFT", True), ("CECUM_AND_COLON", "SIGMOID_COLON", True),
    ("CECUM_AND_COLON", "RECTUM", False), ("LARGE_INTESTINE", "CECUM_AND_COLON", True),
    ("MIDDLE_AND_INNER_EAR_LEFT", "COCHLEA_LEFT", True), ("MIDDLE_AND_INNER_EAR_LEFT", "TYMPANIC_CAVITY_RIGHT", False),
    # declared narrower parts
    ("FEMUR_PROXIMAL", "FEMUR_HEAD_LEFT", True), ("FEMUR_DISTAL", "FEMUR_HEAD", False),
    ("HUMERUS_PROXIMAL_LEFT", "HUMERUS_HEAD_LEFT", True), ("VERTEBRA_L3_ARCH", "VERTEBRA_L3_BODY", False),
    ("PRESACRAL_SPINE_BODY", "VERTEBRA_L3_BODY_CORTICAL", True), ("LIVER_SEGMENT_4", "LIVER_SEGMENT_4A", True),
    ("LIVER_SEGMENT_4", "LIVER_SEGMENT_5", False),
    ("POSTERIOR_CEREBRAL_ARTERY_P1_P2_LEFT", "POSTERIOR_CEREBRAL_ARTERY_P2_LEFT", True),
    ("POSTERIOR_CEREBRAL_ARTERY_P1_P2", "POSTERIOR_CEREBRAL_ARTERY_P3", False),
    ("INTERNAL_CAROTID_ARTERY_INTRACRANIAL_LEFT", "INTERNAL_CAROTID_ARTERY_C5_LEFT", True),
    ("INTERNAL_CAROTID_ARTERY_INTRACRANIAL", "INTERNAL_CAROTID_ARTERY_C2", False),
    ("LATERAL_VENTRICLE_EXCEPT_TEMPORAL_HORN_LEFT", "LATERAL_VENTRICLE_FRONTAL_HORN_LEFT", True),
    ("LATERAL_VENTRICLE_EXCEPT_TEMPORAL_HORN", "LATERAL_VENTRICLE_TEMPORAL_HORN", False),
    # declared narrower tissues
    ("BRAIN_TUMOR", "BRAIN_TUMOR_ENHANCING", True), ("BRAIN_TUMOR", "BRAIN_TUMOR_NECROSIS", True),
    ("BRAIN_TUMOR_NECROSIS_OR_NON_ENHANCING", "BRAIN_TUMOR_NON_ENHANCING", True),
    ("BRAIN_TUMOR", "BRAIN_PERITUMORAL_EDEMA", False), ("BRAIN_TUMOR_AND_EDEMA", "BRAIN_PERITUMORAL_EDEMA", True),
    ("BRAIN_TUMOR_AND_EDEMA", "BRAIN_TUMOR_ENHANCING", True), ("BRAIN_TUMOR_AND_EDEMA", "BRAIN_HEMORRHAGE", False),
    ("BRAIN_WHITE_MATTER", "BRAIN_WHITE_MATTER_LESION", True), ("BRAIN", "BRAIN_HEMORRHAGE", True),
    # findings and devices lie in their structure
    ("KIDNEY_LEFT", "KIDNEY_CYST_LEFT", True), ("KIDNEY_CYST", "KIDNEY", False), ("URINARY_SYSTEM", "KIDNEY_TUMOR", True),
    ("HIP_JOINT_LEFT", "HIP_JOINT_PROSTHESIS_LEFT", True),
    ("LEFT_VENTRICLE", "LEFT_VENTRICULAR_MYOCARDIUM_INFARCT", True),
    # heart, lungs, extra parents
    ("LEFT_VENTRICLE", "LEFT_VENTRICLE_LUMEN", True), ("HEART", "LEFT_VENTRICULAR_MYOCARDIUM", True),
    ("HEART_AND_PERICARDIUM", "PERICARDIUM", True), ("HEART", "PERICARDIUM", False),
    ("LUNG_LEFT", "LUNG_UPPER_LOBE_LEFT", True), ("LUNG", "LUNG_MIDDLE_LOBE_RIGHT", True),
    ("LUNG_LEFT", "LUNG_MIDDLE_LOBE_RIGHT", False), ("THYROID_GLAND_LEFT", "THYROID_LOBE_LEFT", True),
    ("MIDDLE_EAR_LEFT", "MALLEUS_LEFT", True), ("MIDDLE_EAR_LEFT", "MALLEUS_RIGHT", False), ("EAR", "STAPES", True),
    ("TEMPORAL_LOBE_LEFT", "HIPPOCAMPUS_LEFT", True), ("BODY", "UPPER_LIMB_LEFT", True), ("BODY", "LIVER", False),
]


def spec_copy():
    """A copy of the spec the tests may change."""
    s = types.SimpleNamespace(**{k: v for k, v in vars(labelmap_spec).items() if not k.startswith("__")})
    s.TREE, s.ROOTS, s.TISSUES = copy.deepcopy(s.TREE), copy.deepcopy(s.ROOTS), copy.deepcopy(s.TISSUES)
    return s


def node_in(spec, name):
    """The node of a spec copy with this name."""
    todo = [*spec.ROOTS.values(), *(n for nodes in spec.TREE.values() for n in nodes)]
    while todo:
        n = todo.pop()
        if n.name == name:
            return n
        todo.extend(n.children)
    raise KeyError(name)


class TestTheLabelMap(unittest.TestCase):
    def test_labelmap_data_is_what_the_spec_gives(self):
        generated = labelmap_build.render(labelmap_build.build())
        self.assertEqual((PACKAGE / "labelmap_data.py").read_text(encoding="utf-8"), generated,
                         "labelmap_data.py is out of date: run python tools/generate_labelmap.py")

    def test_every_rule_holds(self):
        published, snapshot = labelmap_build.read_published(PACKAGE / "labelmap_published.csv")
        self.assertEqual(labelmap_build.check(labelmap_build.build(), published, snapshot), [])

    def test_names_spell_their_fields(self):
        for label in LM:
            with self.subTest(label=label.name):
                self.assertEqual(label.value, label.name)
                parts = [node_of(label), part_of(label), tissue_of(label), side_of(label)]
                self.assertEqual("_".join(p for p in parts if p), label.name)

    def test_every_label_has_a_kind_a_category_and_a_section(self):
        for label in LM:
            with self.subTest(label=label.name):
                self.assertIn(kind_of(label), {k for k, _ in labelmap_spec.KINDS.values()})
                self.assertIn(category_of(label), ("structure", "tissue", "finding", "device"))
                self.assertIn(section_of(label), SECTIONS)

    def test_examples(self):
        self.assertEqual(node_of(LM.FEMUR_PROXIMAL_CORTICAL_LEFT), "FEMUR")
        self.assertEqual((part_of(LM.FEMUR_PROXIMAL_CORTICAL_LEFT), tissue_of(LM.FEMUR_PROXIMAL_CORTICAL_LEFT),
                          side_of(LM.FEMUR_PROXIMAL_CORTICAL_LEFT)), ("PROXIMAL", "CORTICAL", "LEFT"))
        self.assertEqual((section_of(LM.FEMUR_LEFT), section_of(LM.KIDNEY_CYST_LEFT), section_of(LM.BREAST_IMPLANT)),
                         ("Leg", "Findings", "Devices"))
        self.assertEqual((kind_of(LM.LEFT_ATRIUM), category_of(LM.KIDNEY_CYST_LEFT)), ("heart", "finding"))
        self.assertTrue(is_aggregate(LM.SKULL_WITHOUT_MANDIBLE))
        self.assertIn(LM.CRANIUM, members_of(LM.SKULL_WITHOUT_MANDIBLE))
        self.assertNotIn("LUNG_MIDDLE_LOBE_LEFT", LM.__members__)  # exists on the right only
        self.assertEqual({l.name for l in labels_covered(LM.PATELLA)} >= {"PATELLA_LEFT", "PATELLA_RIGHT"}, True)


class TestCoverage(unittest.TestCase):
    def test_golden_cases(self):
        for a, b, expected in GOLDEN:
            with self.subTest(a=a, b=b):
                self.assertEqual(covers(LM[a], LM[b]), expected)

    def test_covers_is_reflexive_and_transitive_on_a_sample(self):
        rng = random.Random(0)
        names = rng.sample(list(TABLES.labels), 150)
        for a in names:
            self.assertTrue(covers(a, a))
        for a in names:
            below = [b for b in names if covers(a, b)]
            for b in below:
                for c in names:
                    if covers(b, c):
                        with self.subTest(a=a, b=b, c=c):
                            self.assertTrue(covers(a, c))


class TestTheChecksCatchMistakes(unittest.TestCase):
    """Each case breaks a copy of the spec in a way the adversarial reviews found, and asks for the right answer or a
    reported problem."""

    def problems(self, spec, **kw):
        return labelmap_build.check(labelmap_build.build(spec), **kw)

    def test_a_weakened_published_relation_is_caught(self):
        names = ["LIVER", "LIVER_SEGMENT_4", "LIVER_SEGMENT_4A", "BRAIN_TUMOR", "BRAIN_TUMOR_ENHANCING", "SACRUM_S2_S5",
                 "VERTEBRA_S3", "MIDDLE_EAR", "MALLEUS"]
        snapshot = labelmap_build.covers_snapshot(labelmap_build.build(), names)
        spec = spec_copy()
        node_in(spec, "LIVER").part_broader = {}
        problems = self.problems(spec, published=names, published_covers=snapshot)
        self.assertTrue(any("LIVER_SEGMENT_4 covers changed" in p for p in problems), problems)

    def test_a_published_label_cannot_be_renamed(self):
        spec = spec_copy()
        node_in(spec, "HYOID").name = "HYOID_BONE"
        self.assertTrue(any("HYOID" in p and "renamed" in p for p in self.problems(spec, published=["HYOID"])))

    def test_parts_do_not_cross_extra_parents(self):
        spec = spec_copy()
        node_in(spec, "LYMPH_NODES").children.append(
            labelmap_spec.n("MEDIASTINAL_LYMPH_NODES", parts=("ANTERIOR", "STATION_2"), also_under=["MEDIASTINUM"]))
        t = labelmap_build.build(spec)
        self.assertTrue(t.covers("MEDIASTINUM", "MEDIASTINAL_LYMPH_NODES_STATION_2"))
        self.assertFalse(t.covers("MEDIASTINUM_ANTERIOR", "MEDIASTINAL_LYMPH_NODES_ANTERIOR"))

    def test_a_root_declares_narrower_parts(self):
        spec = spec_copy()
        cord = spec.ROOTS["SPINAL_CORD"]
        cord.parts = (*cord.parts, "C5")
        cord.part_broader = {"C5": ["CERVICAL"]}
        t = labelmap_build.build(spec)
        self.assertTrue(t.covers("SPINAL_CORD_CERVICAL", "SPINAL_CORD_C5"))
        self.assertFalse(t.covers("SPINAL_CORD_LUMBAR", "SPINAL_CORD_C5"))

    def test_a_child_lies_in_a_part_of_its_parent(self):
        spec = spec_copy()
        node_in(spec, "AORTA").children.append(labelmap_spec.n("AORTIC_ROOT", in_part="ASCENDING"))
        t = labelmap_build.build(spec)
        self.assertEqual(labelmap_build.check(t), [])
        self.assertTrue(t.covers("AORTA_ASCENDING", "AORTIC_ROOT"))
        self.assertFalse(t.covers("AORTA_ARCH", "AORTIC_ROOT"))

    def test_broken_declarations_are_reported(self):
        cases = {
            "narrower tissues form a cycle": lambda s: s.TISSUES.__setitem__("LESION", ("finding", ["METASTASIS"])),
            "an aggregate contains itself": lambda s: setattr(node_in(s, "GONADS"), "members", ["TESTIS", "OVARY", "GONADS"]),
            "has a cycle": lambda s: setattr(node_in(s, "MIDDLE_EAR"), "also_under", ["AUDITORY_OSSICLES"]),
            "broader LESION is a finding": lambda s: s.TISSUES.__setitem__("STENT", ("device", ["LESION"])),
            "sided members": lambda s: setattr(node_in(s, "GONADS"), "members", ["TESTIS_LEFT", "OVARY_RIGHT"]),
            "not a part of its parent": lambda s: node_in(s, "AORTA").children.append(labelmap_spec.n("AORTIC_ROOT", in_part="NOPE")),
            "nearest common ancestor": lambda s: setattr(node_in(s, "CAROTID_ARTERY"), "members", ["COMMON_CAROTID_ARTERY", "AORTA"]),
        }
        for expected, change in cases.items():
            spec = spec_copy()
            change(spec)
            with self.subTest(expected=expected):
                self.assertTrue(any(expected in p for p in self.problems(spec)), self.problems(spec)[:5])

    def test_a_word_that_spells_an_existing_name_is_refused(self):
        spec = spec_copy()
        spec.TISSUES["CARTILAGE"] = ("tissue", [])
        femur = node_in(spec, "FEMUR")
        femur.tissues = (*femur.tissues, "CARTILAGE")
        femur.part_tissues = (*femur.part_tissues, "DISTAL")
        self.assertTrue(any("duplicate label name FEMUR_DISTAL_CARTILAGE" in p for p in self.problems(spec)))

    def test_only_is_inherited(self):
        spec = spec_copy()
        node_in(spec, "LUNG_MIDDLE_LOBE").children.append(labelmap_spec.n("MEDIAL_SEGMENT_OF_MIDDLE_LOBE"))
        t = labelmap_build.build(spec)
        self.assertEqual([n for n in t.labels if n.startswith("MEDIAL_SEGMENT")], ["MEDIAL_SEGMENT_OF_MIDDLE_LOBE_RIGHT"])
        self.assertEqual(labelmap_build.check(t), [])


@unittest.skipUnless(NODE, "needs node to run the webpage's JavaScript")
class TestTheWebpageAgrees(unittest.TestCase):
    def test_javascript_covers_gives_the_same_answers(self):
        rng = random.Random(1)
        names = list(TABLES.labels)
        pairs = [(a, b) for a, b, _ in GOLDEN]
        pairs += [(rng.choice(names), rng.choice(names)) for _ in range(20000)]
        aggregates = [n for n in names if is_aggregate(n)]
        pairs += [(a, b) for a in aggregates for b in rng.sample(names, 200)]
        pairs += [(b, a) for a in aggregates for b in rng.sample(names, 200)]
        data = structures_data()
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "in.json").write_text(json.dumps({"labelmap": {k: data[k] for k in ("fields", "nodes", "tissueBroader")},
                                                           "pairs": pairs}))
            script = (f"const {{makeCovers}} = require({json.dumps(str(PACKAGE / 'webpage_template' / 'labelmap.js'))});"
                      f"const d = JSON.parse(require('fs').readFileSync({json.dumps(str(Path(folder, 'in.json')))}));"
                      "const c = makeCovers(d.labelmap); console.log(JSON.stringify(d.pairs.map(([a, b]) => c(a, b))));")
            out = subprocess.run([str(NODE), "-e", script], capture_output=True, text=True, check=True).stdout
        js = json.loads(out)
        differ = [(a, b, j) for (a, b), j in zip(pairs, js) if covers(a, b) != j]
        self.assertEqual(differ, [])


if __name__ == "__main__":
    unittest.main()
