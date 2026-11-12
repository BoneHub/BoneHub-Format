"""Find scans by their labels, with the index of a BoneHub-format database.

Build the index first: python -m bonehub_format index /mnt/nas/BoneHub/Harmonized_Databases/MyDatabase
"""

from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/nas/BoneHub/Harmonized_Databases/MyDatabase")

if __name__ == "__main__":
    scans = pd.read_parquet(ROOT / "index" / "scans.parquet")

    # scans with a segmentation of the L4 or L5 vertebra and of a tibia, not cut off by the field of view
    def wanted(row) -> bool:
        labels, truncated = set(row["segmentation"]), set(row["truncated"] if row["truncated"] is not None else [])
        lumbar = {"VERTEBRA_L4", "VERTEBRA_L5"} & labels - truncated
        tibia = {"TIBIA_LEFT", "TIBIA_RIGHT"} & labels - truncated
        return bool(lumbar and tibia)

    selected = scans[scans.apply(wanted, axis=1)]
    # Keep one scan per person. Datasets made from one another (derived_from in index/datasets.parquet) hold the same
    # people, so leave out the scans of a dataset whose parent is selected too.
    datasets = pd.read_parquet(ROOT / "index" / "datasets.parquet")
    parents = {row.dataset_id: set(row.derived_from if row.derived_from is not None else []) for row in datasets.itertuples()}
    chosen = set(selected["dataset_id"])
    selected = selected[[not (parents[dataset_id] & chosen) for dataset_id in selected["dataset_id"]]]
    selected = selected.drop_duplicates(["dataset_id", "subject_id"])
    print(f"Found {len(selected)} people with a complete L4 or L5 vertebra and tibia segmented.")
    print(selected[["scan", "modality", "age", "gender", "license"]].to_string(index=False))
