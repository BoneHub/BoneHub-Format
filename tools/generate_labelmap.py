"""Write bonehub_format/labelmap_data.py (the LabelMap enum) from bonehub_format/labelmap_spec.py.

    python tools/generate_labelmap.py              check the spec and write labelmap_data.py
    python tools/generate_labelmap.py --publish    the same, and freeze the labels of a database release:
                                                   labelmap_published.csv and labelmap_published_covers.json

Nothing is written while a rule of the label map is broken; the problems are printed instead. Once frozen, a published
label keeps its name, and what it covers among published labels never changes.
"""

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "bonehub_format"
DATA = PACKAGE / "labelmap_data.py"
PUBLISHED = PACKAGE / "labelmap_published.csv"


def _load(name: str):
    """A module of the package, loaded on its own: the package itself needs labelmap_data.py, which is being written."""
    spec = importlib.util.spec_from_file_location(f"bonehub_format.{name}", PACKAGE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    sys.modules.setdefault("bonehub_format", type(sys)("bonehub_format"))
    sys.modules["bonehub_format"].__path__ = [str(PACKAGE)]
    _load("labelmap_spec")
    build = _load("labelmap_build")
    tables = build.build()
    published, published_covers = build.read_published(PUBLISHED)
    problems = build.check(tables, published, published_covers)
    if problems:
        print(f"{len(problems)} problems, nothing written:")
        print("\n".join(f"  {p}" for p in problems[:100]))
        return 1
    DATA.write_text(build.render(tables), encoding="utf-8", newline="\n")
    print(f"{DATA.relative_to(ROOT)}: {len(tables.labels)} labels")
    if "--publish" in argv:
        names = list(tables.labels)
        with open(PUBLISHED, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["name"])
            writer.writerows([name] for name in names)
        snapshot = build.covers_snapshot(tables, names)
        PUBLISHED.with_name("labelmap_published_covers.json").write_text(
            json.dumps(snapshot, separators=(",", ":")), encoding="utf-8", newline="\n")
        print(f"froze {len(names)} labels in {PUBLISHED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
