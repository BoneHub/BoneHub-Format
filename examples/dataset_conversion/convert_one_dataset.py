from pathlib import Path

from bonehub_format.custom_dataset_io import SynthRAD2023

database_root = Path("/mnt/nas/BoneHub/Harmonized_Databases/MyDatabase")
data_root = Path("/mnt/nas/BoneHub/Public_Datasets/0006 SynthRAD2023")

# The guard is required: subjects are converted in worker processes that re-import this script.
if __name__ == "__main__":
    SynthRAD2023(dataset_root=data_root).export_to_bonehub_format(
        database_root,
        output_dataset_id=6,  # SynthRAD 2023's BoneHub ID in the registry
        overwrite=True,  # or resume=True to keep the subjects an interrupted run finished
        num_workers=8,
    )
