import unittest
from collections import defaultdict

from bonehub_format import BoneLabelMap
from bonehub_format.labelmap import part_of, side_of, structure_of, tissue_of

LONG_BONES = {4300, 4400, 4500, 7100, 7300, 7400}  # humerus, ulna, radius, femur, tibia, fibula
POSITIONS = {"PROXIMAL": 1, "MIDDLE": 2, "DISTAL": 3}

# Every part scheme partitions its bone; these are the only ones allowed.
SCHEMES = {
    "thirds": {1, 2, 3},  # PROXIMAL SHAFT DISTAL
    "short tubular": {4, 2, 5},  # BASE SHAFT HEAD
    "clavicle": {6, 2, 7},  # MEDIAL SHAFT LATERAL
    "scapula": {6, 7},  # MEDIAL LATERAL
    "rib arcs": {8, 7, 9},  # ANTERIOR LATERAL POSTERIOR
    "vertebra": {10, 11},  # BODY ARCH
    "sternum": {12, 10, 13},  # MANUBRIUM BODY XIPHOID_PROCESS
    "hip bone": {20, 21, 22},  # ILIUM ISCHIUM PUBIS
}


def parts_by_structure():
    parts = defaultdict(set)
    for label in BoneLabelMap:
        if part_of(label.value):
            parts[structure_of(label.value)].add(part_of(label.value))
    return parts


class TestPhalanges(unittest.TestCase):
    def test_every_digit_has_the_right_phalanges(self):
        for where in ("HAND", "FOOT"):
            for digit in range(1, 6):
                expected = ["PROXIMAL", "DISTAL"] if digit == 1 else ["PROXIMAL", "MIDDLE", "DISTAL"]
                present = [p for p in POSITIONS if f"PHALANX_{where}_{digit}_{p}" in BoneLabelMap.__members__]
                with self.subTest(where=where, digit=digit):
                    self.assertEqual(present, expected)

    def test_structure_id_encodes_digit_and_position(self):
        for where, base in (("HAND", 5300), ("FOOT", 8300)):
            for digit in range(1, 6):
                for position, code in POSITIONS.items():
                    name = f"PHALANX_{where}_{digit}_{position}"
                    if name in BoneLabelMap.__members__:
                        with self.subTest(name=name):
                            self.assertEqual(structure_of(BoneLabelMap[name].value), base + digit * 10 + code)


class TestParts(unittest.TestCase):
    def test_every_structure_uses_one_partitioning_scheme(self):
        for structure, parts in parts_by_structure().items():
            with self.subTest(structure=BoneLabelMap(structure * 100000).name):
                self.assertIn(parts, SCHEMES.values())

    def test_a_part_name_has_one_code_everywhere(self):
        code_of = {}
        for label in BoneLabelMap:
            part = part_of(label.value)
            if not part:
                continue
            # the part name is what the structure's whole-bone name is followed by
            base = BoneLabelMap(structure_of(label.value) * 100000).name
            word = label.name[len(base) + 1 :].split("_CORTICAL")[0].split("_TRABECULAR")[0].split("_MEDULLARY")[0]
            word = word.removesuffix("_LEFT").removesuffix("_RIGHT")
            with self.subTest(label=label.name):
                self.assertEqual(code_of.setdefault(word, part), part)

    def test_the_atlas_has_no_body(self):
        self.assertNotIn(2101, parts_by_structure())


class TestTissue(unittest.TestCase):
    def test_medullary_cavity_only_on_long_bones(self):
        for label in BoneLabelMap:
            if tissue_of(label.value) == 3:
                with self.subTest(label=label.name):
                    self.assertIn(structure_of(label.value), LONG_BONES)


class TestSides(unittest.TestCase):
    def test_side_suffix_matches_the_value(self):
        for label in BoneLabelMap:
            expected = 1 if label.name.endswith("_LEFT") else 2 if label.name.endswith("_RIGHT") else 0
            with self.subTest(label=label.name):
                self.assertEqual(side_of(label.value), expected)

    def test_paired_structures_have_both_sides_for_every_label(self):
        values = {label.value for label in BoneLabelMap}
        for label in BoneLabelMap:
            if side_of(label.value) == 1:
                with self.subTest(label=label.name):
                    self.assertIn(label.value - 1, values)  # side not distinguished
                    self.assertIn(label.value + 1, values)  # right

    def test_groups_of_paired_bones_are_paired_too(self):
        for name in (
            "RIBS",
            "COSTAL_CARTILAGES",
            "CARPALS",
            "METATARSALS",
            "PHALANGES_HAND_3",
            "HAND",
            "UPPER_EXTREMITY",
            "SHOULDER_GIRDLE",
            "AUDITORY_OSSICLES",
            "SESAMOIDS_FOOT_1",
        ):
            with self.subTest(group=name):
                self.assertIn(f"{name}_LEFT", BoneLabelMap.__members__)


if __name__ == "__main__":
    unittest.main()
