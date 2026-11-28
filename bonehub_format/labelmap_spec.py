"""The label map as it is written by hand: the tree of structures, their parts and tissues, and the relations a
tree alone cannot say. labelmap_data.py (the LabelMap enum and its tables) is generated from this file:

    python tools/generate_labelmap.py

Edit this file, never labelmap_data.py, and run the generator. A label, once a database is published, keeps its name
and meaning for good: add labels and relations, never rename or remove one (tests/test_labelmap.py checks it against
the published registry).

Every label is a structure of the tree, or a part, tissue or side of one; its name says which:

    NODE[_PART][_TISSUE][_SIDE]      FEMUR_PROXIMAL_CORTICAL_LEFT, VERTEBRA_L3_BODY, KIDNEY_CYST_LEFT, BODY_LESION

Writing a node: n(NAME, *children, side=..., parts=..., tissues=..., ...), see N below.
- side "paired": the node (and its children) exist on both sides: NAME, NAME_LEFT, NAME_RIGHT. "halves": an unpaired
  structure with left and right halves (MANDIBLE_LEFT). "single": a child of a paired node that is not paired.
  only="RIGHT": a structure on one side only (the middle lobe of the lung).
- parts: leaf subdivisions of the node (thirds, body, heads, segments). A part never has parts; a unit that has
  subdivisions of its own, or is an organ unit with its own name (a lobe, a chamber), is a child node instead.
- tissues: what the node's voxels are made of (cortical bone, renal cortex), a finding in it (a cyst, a tumor) or a
  device in it (an implant). They come from TISSUES below.
- members: an aggregate, a union of labels that is no node of the tree (SKULL_WITHOUT_MANDIBLE). It sits under the
  nearest common ancestor of its members (groups across kinds under GROUPS_ACROSS_KINDS).
- part_broader: parts that lie inside other parts of this node and the nodes below it (the femoral head in the
  proximal femur). in_part: the part of its parent a node lies in. also_under: further parents of a node.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The kinds of structure, each the root of a tree, in display order: root -> (kind, webpage section).
#: GROUPS_ACROSS_KINDS holds aggregates whose members are of several kinds; it is no label itself.
KINDS = {
    "SKELETON": ("bone", None),  # the webpage shows bones by region, see BONE_SECTIONS
    "CARTILAGE": ("cartilage", "Cartilage"),
    "LIGAMENTS": ("ligament", "Ligaments"),
    "JOINTS": ("joint", "Joints"),
    "MUSCLES": ("muscle", "Muscles"),
    "HEART_AND_PERICARDIUM": ("heart", "Heart"),
    "ARTERIES": ("artery", "Arteries"),
    "VEINS": ("vein", "Veins"),
    "LYMPHOID_ORGANS": ("lymphoid organ", "Lymphoid organs"),
    "BRAIN": ("brain", "Brain"),
    "SPINAL_CORD": ("spinal cord", "Spinal cord"),
    "CEREBROSPINAL_FLUID": ("CSF space", "CSF spaces"),
    "NERVES": ("nerve", "Nerves"),
    "EYE_AND_ADNEXA": ("eye", "Eye"),
    "EAR": ("ear", "Ear"),
    "RESPIRATORY_SYSTEM": ("respiratory", "Respiratory system"),
    "DIGESTIVE_SYSTEM": ("digestive", "Digestive system"),
    "URINARY_SYSTEM": ("urinary", "Urinary system"),
    "GENITAL_ORGANS": ("genital", "Genital organs"),
    "ENDOCRINE_GLANDS": ("endocrine", "Endocrine glands"),
    "INTEGUMENT": ("integument", "Skin and breast"),
    "ADIPOSE_TISSUE": ("adipose tissue", "Adipose tissue"),
    "BODY_CAVITIES": ("cavity", "Body cavities"),
    "BODY": ("body region", "Body regions"),
    "GROUPS_ACROSS_KINDS": ("group", "Groups across kinds"),
}
#: The webpage's sections of the skeleton: region node -> section (anything else of the skeleton: 'Whole skeleton').
BONE_SECTIONS = {"HEAD_BONES": "Head", "VERTEBRAL_COLUMN": "Spine", "THORAX_BONES": "Thorax",
                 "SHOULDER_AND_ARM_BONES": "Shoulder and arm", "HAND_BONES": "Hand", "PELVIC_GIRDLE": "Pelvis",
                 "LEG_BONES": "Leg", "FOOT_BONES": "Foot"}

#: Tissues: name -> (category, broader tissues). A category is 'tissue' (normal tissue), 'finding' or 'device'. A
#: broader tissue holds the narrower one: a metastasis is a tumor, a tumor is a lesion.
TISSUES = {
    "CORTICAL": ("tissue", []), "TRABECULAR": ("tissue", []), "MEDULLARY_CAVITY": ("tissue", []),
    "BONE_MARROW": ("tissue", []), "CORTEX": ("tissue", []), "MEDULLA": ("tissue", []),
    "GRAY_MATTER": ("tissue", []), "WHITE_MATTER": ("tissue", []), "WALL": ("tissue", []), "LUMEN": ("tissue", []),
    "MUCOSA": ("tissue", []), "FAT": ("tissue", []), "TENDON": ("tissue", []), "VASCULATURE": ("tissue", []),
    "OTHER": ("tissue", []),  # the voxels of the structure no other segment of the file holds
    "LESION": ("finding", []), "CYST": ("finding", ["LESION"]), "TUMOR": ("finding", ["LESION", "TUMOR_AND_EDEMA"]),
    "METASTASIS": ("finding", ["TUMOR"]), "TUMOR_NECROSIS_OR_NON_ENHANCING": ("finding", ["TUMOR"]),
    "TUMOR_ENHANCING": ("finding", ["TUMOR"]),
    "TUMOR_NON_ENHANCING": ("finding", ["TUMOR_NECROSIS_OR_NON_ENHANCING"]),
    "TUMOR_NECROSIS": ("finding", ["TUMOR_NECROSIS_OR_NON_ENHANCING"]),
    "EDEMA": ("finding", []), "PERITUMORAL_EDEMA": ("finding", ["EDEMA", "TUMOR_AND_EDEMA"]),
    "TUMOR_AND_EDEMA": ("finding", []), "HEMORRHAGE": ("finding", []), "INFARCT": ("finding", ["LESION"]),
    "NODULE": ("finding", ["LESION"]), "CALCIFICATION": ("finding", []), "PLAQUE": ("finding", []),
    "THROMBUS": ("finding", []), "EMBOLUS": ("finding", ["THROMBUS"]), "ANEURYSM": ("finding", []),
    "EFFUSION": ("finding", []), "INFILTRATE": ("finding", []), "AIR": ("finding", []), "STONE": ("finding", []),
    "FRACTURE": ("finding", []), "WHITE_MATTER_LESION": ("finding", ["LESION", "WHITE_MATTER"]),
    "RESECTION_CAVITY": ("finding", []),
    "IMPLANT": ("device", []), "PROSTHESIS": ("device", ["IMPLANT"]), "OSTEOSYNTHESIS": ("device", ["IMPLANT"]),
    "STENT": ("device", ["IMPLANT"]),
}

SIDES = ("LEFT", "RIGHT")
SIDE_UNKNOWN = "SIDE_UNKNOWN"  # one side, which is unknown: a label suffix for sources that do not say


@dataclass
class N:
    """A node of the tree: a structure, and the labels made from it."""

    name: str
    children: list = field(default_factory=list)
    side: str | None = None  # None: as the parent ("paired" if it is); "paired", "halves" or "single"
    only: str | None = None  # "LEFT" or "RIGHT": the structure exists on that side only
    parts: tuple = ()
    part_broader: dict = field(default_factory=dict)  # part -> [broader parts] ([] withdraws), for this node and below
    in_part: str | None = None  # the part of its parent this node lies in
    tissues: tuple = ()
    part_tissues: tuple = ()  # parts that get the tissue labels too
    members: list | None = None  # an aggregate: the labels it unites
    also_under: list = field(default_factory=list)  # further parents
    note: str = ""


def n(name, *children, **kw):
    return N(name, list(children), **kw)


BONE = ("CORTICAL", "TRABECULAR")
LONGT = ("CORTICAL", "TRABECULAR", "MEDULLARY_CAVITY")
LONG_BONE = dict(side="paired", parts=("PROXIMAL", "SHAFT", "DISTAL"), tissues=LONGT,
                 part_tissues=("PROXIMAL", "SHAFT", "DISTAL"))
SHORT_TUBULAR = dict(parts=("BASE", "SHAFT", "HEAD"))
POSITIONS = ("PROXIMAL", "MIDDLE", "DISTAL")


def vertebra(name, **kw):
    return n(name, parts=("BODY", "ARCH"), tissues=BONE, part_tissues=("BODY",), **kw)


def phalanges(where):
    return [n(f"PHALANGES_{where}_{d}", *[n(f"{p}_PHALANX_{where}_{d}", **SHORT_TUBULAR)
                                         for p in (("PROXIMAL", "DISTAL") if d == 1 else POSITIONS)])
            for d in range(1, 6)]


TREE = {
    "SKELETON": [
        n("HEAD_BONES",
          n("SKULL",
            n("CRANIUM", n("FRONTAL_BONE"), n("PARIETAL_BONE", side="paired"),
              n("TEMPORAL_BONE", side="paired", parts=("MASTOID_PROCESS",)), n("OCCIPITAL_BONE"),
              n("SPHENOID_BONE"), n("ETHMOID_BONE")),
            n("FACIAL_BONES",
              n("MANDIBLE", side="halves", tissues=BONE), n("MAXILLA", side="paired", tissues=BONE),
              n("ZYGOMATIC_BONE", side="paired"), n("NASAL_BONE", side="paired"),
              n("LACRIMAL_BONE", side="paired"), n("PALATINE_BONE", side="paired"), n("VOMER"),
              n("INFERIOR_NASAL_CONCHA", side="paired")),
            n("SKULL_WITHOUT_MANDIBLE", members=["CRANIUM", "MAXILLA", "ZYGOMATIC_BONE", "NASAL_BONE",
                                                     "LACRIMAL_BONE", "PALATINE_BONE", "VOMER", "INFERIOR_NASAL_CONCHA"]),
            tissues=BONE, note="cranium + facial bones; not the auditory ossicles or the hyoid"),
          n("HYOID"),
          n("AUDITORY_OSSICLES", n("MALLEUS"), n("INCUS"), n("STAPES"), side="paired",
            also_under=["MIDDLE_EAR"])),
        n("VERTEBRAL_COLUMN",
          n("PRESACRAL_SPINE",
            n("CERVICAL_SPINE", n("VERTEBRA_C1", tissues=BONE, note="atlas: no body"),
              *[vertebra(f"VERTEBRA_C{i}") for i in range(2, 8)]),
            n("THORACIC_SPINE", *[vertebra(f"VERTEBRA_T{i}") for i in range(1, 14)]),
            n("LUMBAR_SPINE", *[vertebra(f"VERTEBRA_L{i}") for i in range(1, 7)]),
            parts=("BODY",), note="C1 .. L6, without the sacrum and coccyx"),
          n("SACRUM_COCCYX",
            n("SACRUM", *[vertebra(f"VERTEBRA_S{i}") for i in range(1, 6)],
              n("SACRUM_S2_S5", members=[f"VERTEBRA_S{i}" for i in range(2, 6)]), tissues=BONE),
            n("COCCYX", tissues=BONE)),
          n("VERTEBRAE_AND_SACRUM", members=["PRESACRAL_SPINE", "SACRUM"]),
          ),
        n("THORAX_BONES",
          n("RIBS", *[n(f"RIB_{i}", parts=("ANTERIOR", "LATERAL", "POSTERIOR")) for i in range(1, 14)],
            side="paired"),
          n("STERNUM", parts=("MANUBRIUM", "BODY", "XIPHOID_PROCESS"), tissues=BONE)),
        n("SHOULDER_AND_ARM_BONES",
          n("SHOULDER_GIRDLE", n("CLAVICLE", parts=("MEDIAL", "SHAFT", "LATERAL"), tissues=BONE),
            n("SCAPULA", parts=("MEDIAL", "LATERAL"), tissues=BONE)),
          n("HUMERUS", **{**LONG_BONE, "parts": ("PROXIMAL", "SHAFT", "DISTAL", "HEAD")},
            part_broader={"HEAD": ["PROXIMAL"]}),
          n("ULNA", **LONG_BONE), n("RADIUS", **LONG_BONE), side="paired"),
        n("HAND_BONES",
          n("CARPALS", n("SCAPHOID"), n("LUNATE"), n("TRIQUETRUM"), n("PISIFORM"), n("TRAPEZIUM"),
            n("TRAPEZOID"), n("CAPITATE"), n("HAMATE")),
          n("METACARPALS", *[n(f"METACARPAL_{i}", **SHORT_TUBULAR) for i in range(1, 6)]),
          n("PHALANGES_HAND", *phalanges("HAND")),
          n("SESAMOIDS_HAND", n("SESAMOIDS_HAND_1", n("SESAMOID_HAND_1_MCP_RADIAL"),
                                   n("SESAMOID_HAND_1_MCP_ULNAR"))), side="paired"),
        n("PELVIC_GIRDLE",
          n("HIP_BONE", parts=("ILIUM", "ISCHIUM", "PUBIS"), tissues=BONE, part_tissues=("ILIUM", "ISCHIUM", "PUBIS")),
          side="paired"),
        n("LEG_BONES",
          n("FEMUR", **{**LONG_BONE, "parts": ("PROXIMAL", "SHAFT", "DISTAL", "HEAD")},
            part_broader={"HEAD": ["PROXIMAL"]}),
          n("PATELLA", tissues=BONE), n("TIBIA", **LONG_BONE), n("FIBULA", **LONG_BONE), n("FABELLA"),
          side="paired"),
        n("FOOT_BONES",
          n("TARSALS", n("TALUS", tissues=BONE), n("CALCANEUS", tissues=BONE), n("NAVICULAR"),
            n("CUBOID"), n("MEDIAL_CUNEIFORM"), n("INTERMEDIATE_CUNEIFORM"), n("LATERAL_CUNEIFORM")),
          n("METATARSALS", *[n(f"METATARSAL_{i}", **SHORT_TUBULAR) for i in range(1, 6)]),
          n("PHALANGES_FOOT", *phalanges("FOOT")),
          n("SESAMOIDS_FOOT", n("SESAMOIDS_FOOT_1", n("SESAMOID_FOOT_1_MTP_TIBIAL"),
                                   n("SESAMOID_FOOT_1_MTP_FIBULAR"))), side="paired"),
        n("AXIAL_SKELETON", members=["HEAD_BONES", "VERTEBRAL_COLUMN", "THORAX_BONES"]),
        n("APPENDICULAR_SKELETON", members=["SHOULDER_AND_ARM_BONES", "HAND_BONES", "PELVIC_GIRDLE", "LEG_BONES",
                                                "FOOT_BONES"], side="paired"),
        n("UPPER_EXTREMITY", members=["SHOULDER_AND_ARM_BONES", "HAND_BONES"], side="paired"),
        n("LOWER_EXTREMITY", members=["PELVIC_GIRDLE", "LEG_BONES", "FOOT_BONES"], side="paired"),
        n("BONY_PELVIS", members=["PELVIC_GIRDLE", "SACRUM", "COCCYX"]),
    ],
    "CARTILAGE": [
        n("SPINE_CARTILAGE",
          n("INTERVERTEBRAL_DISCS",
            n("CERVICAL_INTERVERTEBRAL_DISCS", *[n(f"INTERVERTEBRAL_DISC_C{i + 1}_{'T1' if i == 6 else f'C{i + 2}'}")
                                                    for i in range(1, 7)]),
            n("THORACIC_INTERVERTEBRAL_DISCS", *[n(f"INTERVERTEBRAL_DISC_T{i}_T{i + 1}") for i in range(1, 12)],
              n("INTERVERTEBRAL_DISC_T12_L1"), n("INTERVERTEBRAL_DISC_T12_T13"),
              n("INTERVERTEBRAL_DISC_T13_L1")),
            n("LUMBAR_INTERVERTEBRAL_DISCS", *[n(f"INTERVERTEBRAL_DISC_L{i}_L{i + 1}") for i in range(1, 5)],
              n("INTERVERTEBRAL_DISC_L5_S1"), n("INTERVERTEBRAL_DISC_L5_L6"), n("INTERVERTEBRAL_DISC_L6_S1")))),
        n("THORAX_CARTILAGE", n("COSTAL_CARTILAGES", *[n(f"COSTAL_CARTILAGE_{i}") for i in range(1, 11)]),
          side="paired"),
        n("PELVIS_CARTILAGE", n("ACETABULUM_CARTILAGE"), side="paired"),
        n("LEG_CARTILAGE", n("FEMUR_HEAD_CARTILAGE"), n("FEMUR_DISTAL_CARTILAGE"), n("PATELLA_CARTILAGE"),
          n("TIBIA_PLATEAU_MEDIAL_CARTILAGE"), n("TIBIA_PLATEAU_LATERAL_CARTILAGE"), n("TIBIA_DISTAL_CARTILAGE"),
          n("MEDIAL_MENISCUS"), n("LATERAL_MENISCUS"), side="paired"),
        n("FOOT_CARTILAGE", n("TALUS_CARTILAGE"), side="paired"),
    ],
    "LIGAMENTS": [n("LEG_LIGAMENTS", n("ANTERIOR_CRUCIATE_LIGAMENT"), n("POSTERIOR_CRUCIATE_LIGAMENT"),
           n("MEDIAL_COLLATERAL_LIGAMENT"), n("LATERAL_COLLATERAL_LIGAMENT"), n("PATELLAR_LIGAMENT"),
           side="paired")],
    "JOINTS": [n("HEAD_JOINTS", n("TEMPOROMANDIBULAR_JOINT"), side="paired"),
         n("SHOULDER_AND_ARM_JOINTS", n("SHOULDER_JOINT"), side="paired"),
         n("PELVIS_JOINTS", n("HIP_JOINT", tissues=("PROSTHESIS",), side="paired"),
           n("SACROILIAC_JOINT", side="paired"), n("PUBIC_SYMPHYSIS", side="single")),
         n("LEG_JOINTS", n("KNEE_JOINT"), side="paired")],
    "MUSCLES": [
        n("HEAD_MUSCLES", n("EXTRAOCULAR_MUSCLES"), side="paired"),
        n("NECK_MUSCLES", n("PHARYNGEAL_CONSTRICTORS", n("SUPERIOR_PHARYNGEAL_CONSTRICTOR"),
                               n("MIDDLE_PHARYNGEAL_CONSTRICTOR"),
                               n("INFERIOR_PHARYNGEAL_CONSTRICTOR", n("CRICOPHARYNGEUS")), side="single")),
        n("BACK_MUSCLES", n("AUTOCHTHONOUS_BACK_MUSCLES", n("ERECTOR_SPINAE")), side="paired"),
        n("ABDOMINAL_WALL_MUSCLES", n("RECTUS_ABDOMINIS"), side="paired"),
        n("HIP_MUSCLES",
          n("GLUTEAL_MUSCLES", n("GLUTEUS_MAXIMUS"), n("GLUTEUS_MEDIUS"), n("GLUTEUS_MINIMUS"),
            n("TENSOR_FASCIAE_LATAE")),
          n("DEEP_HIP_ROTATORS", n("PIRIFORMIS"), n("GEMELLUS_SUPERIOR"), n("GEMELLUS_INFERIOR"),
            n("OBTURATOR_INTERNUS"), n("OBTURATOR_EXTERNUS"), n("QUADRATUS_FEMORIS")),
          n("ILIOPSOAS", n("ILIACUS"), n("PSOAS_MAJOR")), side="paired"),
        n("LEG_MUSCLES",
          n("THIGH_MUSCLES",
            n("QUADRICEPS_FEMORIS", n("RECTUS_FEMORIS"), n("VASTUS_LATERALIS"), n("VASTUS_MEDIALIS"),
              n("VASTUS_INTERMEDIUS")),
            n("THIGH_ADDUCTORS", n("ADDUCTOR_LONGUS"), n("ADDUCTOR_BREVIS"), n("ADDUCTOR_MAGNUS"),
              n("GRACILIS"), n("PECTINEUS")),
            n("HAMSTRINGS", n("BICEPS_FEMORIS", parts=("LONG_HEAD", "SHORT_HEAD")), n("SEMITENDINOSUS"),
              n("SEMIMEMBRANOSUS")),
            n("SARTORIUS")),
          n("LOWER_LEG_MUSCLES",
            n("ANTERIOR_LEG_MUSCLES", n("TIBIALIS_ANTERIOR"), n("EXTENSOR_DIGITORUM_LONGUS"),
              n("EXTENSOR_HALLUCIS_LONGUS")),
            n("FIBULAR_MUSCLES", n("FIBULARIS_LONGUS"), n("FIBULARIS_BREVIS")),
            n("SUPERFICIAL_POSTERIOR_LEG_MUSCLES", n("GASTROCNEMIUS", parts=("MEDIAL_HEAD", "LATERAL_HEAD")),
              n("SOLEUS"), n("PLANTARIS")),
            n("DEEP_POSTERIOR_LEG_MUSCLES", n("TIBIALIS_POSTERIOR"), n("FLEXOR_DIGITORUM_LONGUS"),
              n("FLEXOR_HALLUCIS_LONGUS"), n("POPLITEUS"))), side="paired"),
    ],
    "HEART_AND_PERICARDIUM": [n("HEART",
           n("LEFT_ATRIUM", n("LEFT_ATRIAL_APPENDAGE"), tissues=("LUMEN",)),
           n("RIGHT_ATRIUM", n("RIGHT_ATRIAL_APPENDAGE"), tissues=("LUMEN",)),
           n("LEFT_VENTRICLE", n("LEFT_VENTRICULAR_MYOCARDIUM", tissues=("INFARCT",)), tissues=("LUMEN",)),
           n("RIGHT_VENTRICLE", tissues=("LUMEN",)),
           n("HEART_ATRIA", members=["LEFT_ATRIUM", "RIGHT_ATRIUM"])),
         n("PERICARDIUM")],
    "ARTERIES": [
        n("AORTA", parts=("ASCENDING", "ARCH", "DESCENDING_THORACIC", "ABDOMINAL")),
        n("PULMONARY_ARTERY", n("PULMONARY_TRUNK", side="single"), side="halves"),
        n("CORONARY_ARTERIES", n("LEFT_CORONARY_ARTERY"), n("RIGHT_CORONARY_ARTERY")),
        n("HEAD_AND_NECK_ARTERIES", n("BRACHIOCEPHALIC_TRUNK", side="single"), n("COMMON_CAROTID_ARTERY"),
          n("INTERNAL_CAROTID_ARTERY", parts=(*(f"C{k}" for k in range(1, 8)), "INTRACRANIAL"),
            part_broader={f"C{k}": ["INTRACRANIAL"] for k in range(4, 8)}), n("EXTERNAL_CAROTID_ARTERY"),
          n("VERTEBRAL_ARTERY"), n("SUPERFICIAL_TEMPORAL_ARTERY"), n("MAXILLARY_ARTERY"),
          n("MIDDLE_MENINGEAL_ARTERY"),
          n("CAROTID_ARTERY", members=["COMMON_CAROTID_ARTERY", "INTERNAL_CAROTID_ARTERY", "EXTERNAL_CAROTID_ARTERY"]),
          side="paired"),
        n("BRAIN_ARTERIES",
          n("BASILAR_ARTERY", side="single"), n("ANTERIOR_CEREBRAL_ARTERY", parts=("A1", "A2", "A3", "A4", "A5", "A1_A2")),
          n("MIDDLE_CEREBRAL_ARTERY", parts=("M1", "M2", "M3", "M4")),
          n("POSTERIOR_CEREBRAL_ARTERY", parts=("P1", "P2", "P3", "P4", "P1_P2", "P3_P4")),
          n("ANTERIOR_COMMUNICATING_ARTERY", side="single"), n("POSTERIOR_COMMUNICATING_ARTERY"),
          n("SUPERIOR_CEREBELLAR_ARTERY"), n("ANTERIOR_INFERIOR_CEREBELLAR_ARTERY"),
          n("POSTERIOR_INFERIOR_CEREBELLAR_ARTERY"), n("ANTERIOR_CHOROIDAL_ARTERY"), n("OPHTHALMIC_ARTERY"),
          n("MEDIAN_ARTERY_OF_CORPUS_CALLOSUM", side="single", parts=("A2", "A3")),
          side="paired",
          part_broader={"A1": ["A1_A2"], "A2": ["A1_A2"], "P1": ["P1_P2"], "P2": ["P1_P2"], "P3": ["P3_P4"], "P4": ["P3_P4"]}),
        n("UPPER_LIMB_ARTERIES", n("SUBCLAVIAN_ARTERY"), side="paired"),
        n("ABDOMINAL_ARTERIES", n("CELIAC_TRUNK")),
        n("PELVIC_ARTERIES", n("ILIAC_ARTERY", n("COMMON_ILIAC_ARTERY"), n("INTERNAL_ILIAC_ARTERY"),
                                  n("EXTERNAL_ILIAC_ARTERY")), side="paired"),
    ],
    "VEINS": [n("SUPERIOR_VENA_CAVA"), n("INFERIOR_VENA_CAVA"), n("PULMONARY_VEIN", side="paired"),
         n("PORTAL_VENOUS_SYSTEM", n("PORTAL_VEIN"), n("SPLENIC_VEIN"), n("SUPERIOR_MESENTERIC_VEIN"),
           n("PORTAL_VEIN_AND_SPLENIC_VEIN", members=["PORTAL_VEIN", "SPLENIC_VEIN"])),
         n("HEAD_AND_NECK_VEINS", n("BRACHIOCEPHALIC_VEIN"), n("INTERNAL_JUGULAR_VEIN"), side="paired"),
         n("INTRACRANIAL_VEINS", n("DURAL_VENOUS_SINUSES")),
         n("PELVIC_VEINS", n("ILIAC_VEIN", n("COMMON_ILIAC_VEIN"), n("INTERNAL_ILIAC_VEIN"),
                                n("EXTERNAL_ILIAC_VEIN")), side="paired")],
    "LYMPHOID_ORGANS": [n("SPLEEN"), n("THYMUS"),
         n("LYMPH_NODES", n("CERVICAL_LYMPH_NODES", side="halves", tissues=("METASTASIS",)), tissues=("METASTASIS",))],
    "BRAIN": [n("CEREBRAL_HEMISPHERE",
           n("FRONTAL_LOBE"),
           n("PARIETAL_LOBE"),
           n("TEMPORAL_LOBE", n("HIPPOCAMPUS"), n("AMYGDALA")),
           n("OCCIPITAL_LOBE"),
           n("INSULA"),
           n("LIMBIC_LOBE"),
           n("BASAL_NUCLEI", n("CAUDATE_NUCLEUS"), n("PUTAMEN"), n("GLOBUS_PALLIDUS"),
             n("NUCLEUS_ACCUMBENS")), side="paired"),
         n("DIENCEPHALON", n("THALAMUS", side="paired"), n("OPTIC_CHIASM")),
         n("BRAINSTEM", n("MIDBRAIN", n("SUBSTANTIA_NIGRA", side="paired")), n("PONS"), n("MEDULLA_OBLONGATA")),
         n("CEREBELLUM", n("CEREBELLAR_HEMISPHERE", side="paired"), n("VERMIS")),
         n("CORPUS_CALLOSUM")],
    "SPINAL_CORD": [], "CEREBROSPINAL_FLUID": [n("VENTRICLES_OF_BRAIN",
                   n("LATERAL_VENTRICLE", side="paired", parts=("FRONTAL_HORN", "CENTRAL_PART", "OCCIPITAL_HORN", "TEMPORAL_HORN", "EXCEPT_TEMPORAL_HORN"),
                     part_broader={p: ["EXCEPT_TEMPORAL_HORN"] for p in ("FRONTAL_HORN", "CENTRAL_PART", "OCCIPITAL_HORN")}),
                   n("THIRD_VENTRICLE"), n("FOURTH_VENTRICLE"))],
    "NERVES": [n("CRANIAL_NERVES", n("OPTIC_NERVE"), side="paired")],
    "EYE_AND_ADNEXA": [n("EYEBALL", n("LENS"), parts=("ANTERIOR_SEGMENT", "POSTERIOR_SEGMENT")), n("LACRIMAL_GLAND")],
    "EAR": [n("MIDDLE_EAR", n("TYMPANIC_CAVITY"), n("AUDITORY_TUBE", parts=("BONY_PART", "CARTILAGINOUS_PART"))),
         n("INTERNAL_EAR", n("COCHLEA"), n("VESTIBULE"), n("SEMICIRCULAR_CANALS"),
           n("VESTIBULE_AND_SEMICIRCULAR_CANALS", members=["VESTIBULE", "SEMICIRCULAR_CANALS"])),
         n("INTERNAL_ACOUSTIC_MEATUS"),
         n("MIDDLE_AND_INNER_EAR", members=["MIDDLE_EAR", "INTERNAL_EAR", "INTERNAL_ACOUSTIC_MEATUS"], side="paired")],
    "RESPIRATORY_SYSTEM": [n("LARYNX", n("ARYTENOID_CARTILAGE", side="paired"), parts=("SUPRAGLOTTIS", "GLOTTIS", "SUBGLOTTIS")),
         n("TRACHEOBRONCHIAL_TREE", n("TRACHEA"), n("BRONCHI", side="halves")),
         n("LUNG", n("LUNG_UPPER_LOBE"), n("LUNG_MIDDLE_LOBE", only="RIGHT"),
           n("LUNG_LOWER_LOBE"), side="paired", tissues=("VASCULATURE", "INFILTRATE")),
         n("PLEURA", side="paired")],
    "DIGESTIVE_SYSTEM": [n("MOUTH", n("ORAL_CAVITY"), n("LIPS")),
         n("SALIVARY_GLANDS", n("PAROTID_GLAND"), n("SUBMANDIBULAR_GLAND"), side="paired"),
         n("PHARYNX", n("NASOPHARYNX", tissues=("TUMOR",)), n("OROPHARYNX"), n("HYPOPHARYNX")),
         n("ESOPHAGUS", parts=("CERVICAL", "THORACIC", "ABDOMINAL")), n("STOMACH"),
         n("SMALL_INTESTINE", n("DUODENUM"), n("JEJUNUM"), n("ILEUM"),
           n("JEJUNUM_AND_ILEUM", members=["JEJUNUM", "ILEUM"])),
         n("LARGE_INTESTINE", n("CECUM"),
           n("COLON", n("ASCENDING_COLON"), n("TRANSVERSE_COLON"), n("DESCENDING_COLON"),
             n("SIGMOID_COLON")), n("RECTUM"), n("ANAL_CANAL"),
           n("CECUM_AND_COLON", members=["CECUM", "COLON"])),
         n("LIVER", parts=("SEGMENT_1", "SEGMENT_2", "SEGMENT_3", "SEGMENT_4", "SEGMENT_4A", "SEGMENT_4B",
                              "SEGMENT_5", "SEGMENT_6", "SEGMENT_7", "SEGMENT_8"),
           part_broader={"SEGMENT_4A": ["SEGMENT_4"], "SEGMENT_4B": ["SEGMENT_4"]}, tissues=("VASCULATURE", "TUMOR")),
         n("GALLBLADDER"), n("PANCREAS", parts=("HEAD", "BODY", "TAIL")),
         n("BOWEL", members=["SMALL_INTESTINE", "LARGE_INTESTINE"])],
    "URINARY_SYSTEM": [n("KIDNEY", side="paired", tissues=("CORTEX", "MEDULLA", "CYST", "TUMOR")),
         n("URETER", side="paired"), n("URINARY_BLADDER"), n("URETHRA")],
    "GENITAL_ORGANS": [n("MALE_GENITAL_ORGANS", n("PROSTATE", parts=("PERIPHERAL_ZONE", "TRANSITION_ZONE", "CENTRAL_ZONE")),
           n("SEMINAL_VESICLE", side="paired"), n("TESTIS", side="paired")),
         n("FEMALE_GENITAL_ORGANS", n("UTERUS", n("UTERINE_CERVIX")), n("OVARY", side="paired")),
         n("GONADS", members=["TESTIS", "OVARY"], side="paired"),
         n("PROSTATE_OR_UTERUS", members=["PROSTATE", "UTERUS"],
           note="for sources that do not say which; resolved by sex where it is known")],
    "ENDOCRINE_GLANDS": [n("PITUITARY_GLAND"), n("THYROID_GLAND", n("THYROID_LOBE", side="paired"), n("THYROID_ISTHMUS"), side="halves"),
         n("ADRENAL_GLAND", side="paired", tissues=("CORTEX", "MEDULLA"))],
    "INTEGUMENT": [n("SKIN"), n("BREAST", n("MAMMARY_GLAND"), side="paired", tissues=("IMPLANT",))],
    "ADIPOSE_TISSUE": [n("SUBCUTANEOUS_ADIPOSE_TISSUE"), n("VISCERAL_ADIPOSE_TISSUE"), n("INTERMUSCULAR_ADIPOSE_TISSUE"),
         n("TORSO_ADIPOSE_TISSUE", members=["VISCERAL_ADIPOSE_TISSUE", "INTERMUSCULAR_ADIPOSE_TISSUE"])],
    "BODY_CAVITIES": [n("CRANIAL_CAVITY"), n("SPINAL_CANAL"),
         n("THORACIC_CAVITY", n("PLEURAL_CAVITY", side="paired", tissues=("EFFUSION", "AIR")),
           n("MEDIASTINUM", parts=("SUPERIOR", "ANTERIOR", "MIDDLE", "POSTERIOR"), tissues=("OTHER",)),
           n("PERICARDIAL_CAVITY", tissues=("EFFUSION",))),
         n("ABDOMINOPELVIC_CAVITY", n("ABDOMINAL_CAVITY"), n("PELVIC_CAVITY"),
           n("BOWEL_SPACE", note="the space the bowel loops take up (the 'bowel bag' of radiotherapy)"), tissues=("OTHER",))],
    "BODY": [n("HEAD", n("FACE", n("CHEEK", side="paired", tissues=("MUCOSA",))), n("SCALP")), n("NECK"),
         n("TRUNK", n("THORAX"), n("ABDOMEN"), n("PELVIS")),
         n("UPPER_LIMB", side="paired"), n("LOWER_LIMB", side="paired"),
         n("LIMBS", members=["UPPER_LIMB", "LOWER_LIMB"], side="paired")],
    "GROUPS_ACROSS_KINDS": [n("HEAD_AND_NECK_GLANDS", members=["PAROTID_GLAND", "SUBMANDIBULAR_GLAND", "THYROID_GLAND"],
           note="CADS 'Gland structure': thyroid, parotid and submandibular glands (checked in the data)"),
         n("HEAD_VESSELS", side="paired", members=["BRAIN_ARTERIES", "INTERNAL_CAROTID_ARTERY_INTRACRANIAL", "INTRACRANIAL_VEINS"],
           note="CADS 'Vascular structure of head': intracranial arteries and the dural sinuses (checked in the data)")],
}

#: What the kind roots hold themselves (they are nodes too).
ROOTS = {
    "LIGAMENTS": n("LIGAMENTS", side="paired"),
    "HEART_AND_PERICARDIUM": n("HEART_AND_PERICARDIUM", tissues=("OTHER",)),
    "BRAIN": n("BRAIN", tissues=("GRAY_MATTER", "WHITE_MATTER", "TUMOR", "TUMOR_AND_EDEMA", "TUMOR_ENHANCING",
                                "TUMOR_NON_ENHANCING", "TUMOR_NECROSIS", "TUMOR_NECROSIS_OR_NON_ENHANCING",
                                "PERITUMORAL_EDEMA", "HEMORRHAGE", "WHITE_MATTER_LESION", "RESECTION_CAVITY")),
    "SPINAL_CORD": n("SPINAL_CORD", parts=("CERVICAL", "THORACIC", "LUMBAR", "SACRAL"), tissues=("GRAY_MATTER", "WHITE_MATTER")),
    "EYE_AND_ADNEXA": n("EYE_AND_ADNEXA", side="paired"),
    "EAR": n("EAR", side="paired"),
    "BODY": n("BODY", tissues=("LESION",)),
}
