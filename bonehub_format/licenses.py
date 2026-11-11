"""Licenses a dataset or a scan can be under.

The `license` of dataset.json and scan.json is one of the SPDX identifiers in LICENSES, or RESTRICTED. RESTRICTED
marks data under a custom license with restricted access: it is converted like any other data, but it stays out of
the published dataset, and the dataset card says where to obtain it.
"""

RESTRICTED = "restricted"

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


def license_name(value: str) -> str:
    if value == RESTRICTED:
        return "Restricted (custom license)"
    return LICENSES[value][0]


def license_url(value: str) -> str | None:
    return None if value == RESTRICTED else LICENSES[value][1]
