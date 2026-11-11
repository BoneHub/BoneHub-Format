from pathlib import Path

from bonehub_format.custom_dataset_io import SynthRAD2023

bonehub_dataset_root = Path("Z:/BoneHub/BoneHub_Dataset")
data_root = Path("Z:/BoneHub/Public_Datasets/006 SynthRAD2023")

# The guard is required: subjects are converted in worker processes that re-import this script.
if __name__ == "__main__":
    SynthRAD2023(dataset_root=data_root).export_to_bonehub_format(
        bonehub_dataset_root,
        output_dataset_id=10,
        overwrite=True,  # or resume=True to keep the subjects an interrupted run finished
        num_workers=8,
    )
