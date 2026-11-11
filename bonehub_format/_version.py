"""Versions of the package and of the data structure it reads and writes."""

__version__ = "1.0.0"

#: Version of the BoneHub format (major.minor), recorded in every dataset.json.
#: A minor version adds something that datasets of an older minor version still satisfy, such as a new optional
#: field or a new kind of data in a scan folder. A major version changes or removes something, so that datasets of an
#: older major version are no longer valid.
SCHEMA_VERSION = "1.0"


def _major_minor(version: str) -> tuple[int, int]:
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


def is_compatible_schema_version(version: str | None) -> bool:
    """True if this package can read a dataset written in `version` of the format: the same major version, and a
    minor version no newer than SCHEMA_VERSION."""
    if not version:
        return False
    try:
        major, minor = _major_minor(version)
    except ValueError:
        return False
    current_major, current_minor = _major_minor(SCHEMA_VERSION)
    return major == current_major and minor <= current_minor
