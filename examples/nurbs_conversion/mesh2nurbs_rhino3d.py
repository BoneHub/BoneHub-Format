"""Convert the meshes of BoneHub-format datasets to NURBS surfaces (IGES format) using Rhino3D.

For every scan of each given dataset, each mesh in its mesh/ folder (or mesh.zip) is converted with the
`mesh2nurbs-rhino3d` CLI and saved as nurbs/<LABEL>.iges; then scan.json lists the new NURBS surfaces, with the
origin derived:mesh2nurbs-rhino3d@<version>, and nurbs.zip is brought up to date. The quad remesh edge length is
chosen per bone from its body region (see QUADREMESH_LENGTH_PER_REGION and QUADREMESH_LENGTH_PER_LABEL below).

Requires the `mesh2nurbs-rhino3d` CLI on PATH, e.g. by activating its conda env first (with bonehub-format installed
in it):
    conda activate mesh2nurbs-rhino3d

Usage:
    python mesh2nurbs_rhino3d.py 36                     # convert all meshes of dataset 36 (VSD Bone Reconstruction)
    python mesh2nurbs_rhino3d.py 36 27 44               # several datasets
    python mesh2nurbs_rhino3d.py 36 --subjects 1 2      # only some subjects
    python mesh2nurbs_rhino3d.py 36 --overwrite         # regenerate existing NURBS surfaces
    python mesh2nurbs_rhino3d.py 36 --dry-run           # print the commands without running them
    python mesh2nurbs_rhino3d.py 36 --root "/mnt/nas/BoneHub/Harmonized_Databases/MyDatabase" --rhino-path "C:/Program Files/Rhino 8/System/Rhino.exe"

Run `python mesh2nurbs_rhino3d.py --help` for all options.
"""

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from bonehub_format import BoneLabelMap, Database, layout, pack_scan, validate
from bonehub_format.labelmap import structure_of

DEFAULT_ROOT = Path("/mnt/nas/BoneHub/Harmonized_Databases/MyDatabase")

CLI_COMMAND = "mesh2nurbs-rhino3d"
DEFAULT_PREPROCESSING_STEPS = ["remove-isolated-islands"]

# Quad remesh edge length (mm) per body region, the first digit of the label's structure id
# (see BoneLabelMap): 1 head, 2 spine, 3 thorax, 4 shoulder+arm, 5 hand, 6 pelvis, 7 leg, 8 foot.
QUADREMESH_LENGTH_PER_REGION = {
    1: 3.0,
    2: 2.0,
    3: 2.0,
    4: 3.0,
    5: 1.0,
    6: 3.0,
    7: 3.0,
    8: 1.0,
}
# Bones that need a different edge length than the rest of their region
QUADREMESH_LENGTH_PER_LABEL = {"PATELLA_LEFT": 2.0, "PATELLA_RIGHT": 2.0}
DEFAULT_QUADREMESH_LENGTH = 2.0


def tool_version() -> str:
    try:
        return importlib.metadata.version(CLI_COMMAND)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def quadremesh_length(label: str) -> float:
    if label in QUADREMESH_LENGTH_PER_LABEL:
        return QUADREMESH_LENGTH_PER_LABEL[label]
    region = structure_of(BoneLabelMap[label].value) // 1000
    return QUADREMESH_LENGTH_PER_REGION.get(region, DEFAULT_QUADREMESH_LENGTH)


def build_command(input_path: Path, label: str, preprocessing_steps: list, rhino_path: str | None) -> list:
    cmd = [
        CLI_COMMAND,
        "--input-path",
        str(input_path),
        "--quadremesh-length",
        str(quadremesh_length(label)),
        "--preprocessing-steps",
        *preprocessing_steps,
        "--output-filetype",
        "iges",
    ]
    if rhino_path:
        cmd += ["--rhino-path", rhino_path]
    return cmd


def convert_dataset(root: Path, dataset_id: int, args: argparse.Namespace) -> list:
    """Convert the meshes of one dataset and return the meshes that failed."""
    dataset = Database(root).dataset(dataset_id)
    print(f"\n=== Dataset {dataset_id}: {dataset.path} ===", flush=True)
    origin = f"derived:{CLI_COMMAND}@{tool_version()}"

    failures = []
    for subject in dataset.subjects():
        if args.subjects and subject.id not in args.subjects:
            continue
        for scan in subject.scans():
            meshes = scan.model_files("mesh")
            if not meshes:
                print(f"[skip] {scan.key}: no meshes")
                continue
            nurbs_folder = layout.model_folder(scan.path, "nurbs")
            converted = []
            for label, files in meshes.items():
                output_path = nurbs_folder / f"{label}.iges"
                if output_path.exists() and not args.overwrite:
                    print(f"NURBS surface already exists at `{output_path}`. Skipping generation.")
                    converted.append(label)
                    continue

                # The CLI writes <stem>.iges next to the input mesh, so run it on a copy in a temporary folder
                # rather than writing into the scan's mesh folder (the mesh may also be inside mesh.zip).
                with tempfile.TemporaryDirectory() as tmp:
                    input_path = Path(tmp) / f"{label}.stl"
                    result_path = input_path.with_suffix(".iges")
                    cmd = build_command(input_path, label, args.preprocessing_steps, args.rhino_path)
                    print(f"Processing {scan.key} {label} ...")
                    print(f"$ {subprocess.list2cmdline(cmd)}", flush=True)
                    if args.dry_run:
                        continue

                    input_path.write_bytes(files[0].read_bytes())
                    returncode = subprocess.run(cmd, cwd=tmp).returncode
                    if returncode != 0 or not result_path.exists():
                        print(f"WARNING: Failed to create NURBS surface for {scan.key} {label} (exit code {returncode}).")
                        failures.append(f"{scan.key} {label}")
                        continue
                    nurbs_folder.mkdir(exist_ok=True)
                    shutil.copyfile(result_path, output_path)
                    print(f"Successfully created NURBS surface at `{output_path}`.")
                    converted.append(label)

            if args.dry_run or not converted:
                continue
            info = scan.info
            origins = dict(info.nurbs or {})
            origins.update({label: origins.get(label, origin) if not args.overwrite else origin for label in converted})
            info.set_labels("nurbs", list(origins), origins)
            pack_scan(scan.path)
            info.save(scan.path / layout.SCAN_JSON)

    if not args.dry_run:
        problems = [problem for problem in validate(root) if dataset.path.name in problem]
        print("Dataset check passed." if not problems else "WARNING: the dataset check found problems:")
        for problem in problems:
            print(f"  - {problem}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_ids", nargs="+", type=int, help="IDs of the datasets to convert")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root folder of the database (default: {DEFAULT_ROOT})",
    )
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to convert (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate NURBS surfaces that already exist")
    parser.add_argument(
        "--preprocessing-steps",
        nargs="+",
        default=DEFAULT_PREPROCESSING_STEPS,
        help=f"Preprocessing steps passed to {CLI_COMMAND} (default: {' '.join(DEFAULT_PREPROCESSING_STEPS)})",
    )
    parser.add_argument("--rhino-path", help=f"Path to Rhino.exe (default: the {CLI_COMMAND} default)")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    args = parser.parse_args()

    if not args.dry_run and shutil.which(CLI_COMMAND) is None:
        print(f"'{CLI_COMMAND}' not found on PATH. Activate the conda env first: conda activate mesh2nurbs-rhino3d")
        return 1

    failures = []
    for dataset_id in args.dataset_ids:
        failures += convert_dataset(args.root, dataset_id, args)

    if failures:
        print(f"\nFailed to convert {len(failures)} mesh(es):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
