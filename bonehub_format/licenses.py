"""Licenses the data of a dataset can be under.

A license is given per kind of data (LICENSE_KINDS: the image, the segmentation, the meshes, the NURBS surfaces, the
landmarks), for
each source in dataset.json, for a subset of a source, or for one file in its JSON file (see metadata.Licenses). Each is one of
the SPDX identifiers in LICENSES, or RESTRICTED. RESTRICTED marks data under a custom license with restricted access:
it is converted like any other data, but its files stay out of the published database, and the dataset card says
where to obtain them. The kinds of one scan can differ, e.g. a restricted image with an open segmentation.
"""

RESTRICTED = "restricted"

#: The kinds of data a license is given for.
LICENSE_KINDS: tuple[str, ...] = ("image", "segmentation", "mesh", "nurbs", "landmarks")

_CC_ELEMENTS = {"BY": "Attribution", "SA": "ShareAlike", "NC": "NonCommercial", "ND": "NoDerivatives"}
_CC_VARIANTS = ("BY", "BY-SA", "BY-NC", "BY-NC-SA", "BY-ND", "BY-NC-ND")
_CC_VERSIONS = ("2.0", "2.5", "3.0", "4.0")


def _creative_commons() -> dict[str, tuple[str, str]]:
    licenses = {}
    for variant in _CC_VARIANTS:
        words = "-".join(_CC_ELEMENTS[element] for element in variant.split("-"))
        for version in _CC_VERSIONS:
            licenses[f"CC-{variant}-{version}"] = (
                f"Creative Commons {words} {version}",
                f"https://creativecommons.org/licenses/{variant.lower()}/{version}/",
            )
    return licenses


#: SPDX identifier -> (name, URL)
LICENSES: dict[str, tuple[str, str]] = {
    "CC0-1.0": ("Creative Commons Zero 1.0 Universal", "https://creativecommons.org/publicdomain/zero/1.0/"),
    **_creative_commons(),
    "ODbL-1.0": ("Open Data Commons Open Database License 1.0", "https://opendatacommons.org/licenses/odbl/1-0/"),
    "ODC-By-1.0": ("Open Data Commons Attribution License 1.0", "https://opendatacommons.org/licenses/by/1-0/"),
    "PDDL-1.0": (
        "Open Data Commons Public Domain Dedication and License 1.0",
        "https://opendatacommons.org/licenses/pddl/1-0/",
    ),
    "MIT": ("MIT License", "https://opensource.org/license/mit"),
    "Apache-2.0": ("Apache License 2.0", "https://www.apache.org/licenses/LICENSE-2.0"),
    "GPL-3.0-only": ("GNU General Public License v3.0 only", "https://www.gnu.org/licenses/gpl-3.0.html"),
}


def check_license(value: str) -> str:
    """Return `value` if it is a known SPDX identifier or RESTRICTED, else raise ValueError."""
    if value != RESTRICTED and value not in LICENSES:
        raise ValueError(
            f"Unknown license '{value}'. Use an SPDX identifier such as 'CC-BY-4.0' or 'CC-BY-NC-SA-4.0' (see "
            f"bonehub_format.licenses.LICENSES), or '{RESTRICTED}' for a custom license with restricted access."
        )
    return value


def is_restricted(value: str | None) -> bool:
    return value == RESTRICTED


def is_publishable(value: str | None) -> bool:
    """True for data that may be published: its license is known and not restricted."""
    return value is not None and value != RESTRICTED


def license_name(value: str) -> str:
    if value == RESTRICTED:
        return "Restricted (custom license)"
    return LICENSES[value][0]


def license_url(value: str) -> str | None:
    return None if value == RESTRICTED else LICENSES[value][1]
