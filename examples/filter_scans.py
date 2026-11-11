"""Find scans by their labels, with the index of a BoneHub-format root.

Build the index first: python -m bonehub_format index Z:/BoneHub/BoneHub_Dataset
"""

from pathlib import Path

import pandas as pd

ROOT = Path("Z:/BoneHub/BoneHub_Dataset")

if __name__ == "__main__":
    scans = pd.read_parquet(ROOT / "index" / "scans.parquet")

    # scans with a segmentation of the L4 or L5 vertebra and of a tibia, not cut off by the field of view
    def wanted(row) -> bool:
        labels, truncated = set(row["segmentation"]), set(row["truncated"] if row["truncated"] is not None else [])
        lumbar = {"VERTEBRA_L4", "VERTEBRA_L5"} & labels - truncated
        tibia = {"TIBIA_LEFT", "TIBIA_RIGHT"} & labels - truncated
        return bool(lumbar and tibia)

    selected = scans[scans.apply(wanted, axis=1)]
    # Some source datasets reuse each other's images: keep one scan per person.
    selected = selected.drop_duplicates("subject_group")
    print(f"Found {len(selected)} people with a complete L4 or L5 vertebra and tibia segmented.")
    print(selected[["scan", "modality", "age", "gender", "license"]].to_string(index=False))
