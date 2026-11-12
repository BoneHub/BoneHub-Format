"""Convert the BoneHub public datasets listed in DATASET_JOBS, several at a time.

Run from the repository root with the package installed (pip install -e ".[converter]")
and the MAX_SUBJECTS_FOR_TESTING environment variable unset; see --help for all options.
Each dataset writes a log into <output-root>/logs, naming any subject that failed with its traceback;
the other subjects are still converted. The run's own log (logs/conversion_*.log) ends with a
summary of which datasets succeeded and which failed.

Convert everything again (sized for 32 GB of RAM):

    python examples/dataset_conversion/convert_all_datasets_parallel.py --output-root /mnt/nas/BoneHub/Harmonized_Databases/MyDatabase --overwrite --max-workers 3 --subject-workers 4

Convert only the subjects that failed (or were not reached) in an earlier run:

    python examples/dataset_conversion/convert_all_datasets_parallel.py --output-root /mnt/nas/BoneHub/Harmonized_Databases/MyDatabase --resume --datasets vsd_lower_extremities_seg

Afterwards, get the database ready to upload (see the README):

    python -m bonehub_format prepare /mnt/nas/BoneHub/Harmonized_Databases/MyDatabase --repo-id BoneHub/MyDatabase --title "My Database"
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import bonehub_format.custom_dataset_io as custom_dataset_io
from bonehub_format import layout


@dataclass(frozen=True)
class DatasetJob:
    name: str
    dataset_id: int  # the dataset's BoneHub ID in the registry
    dataset_class: type
    source_root: Path


DATASET_JOBS: tuple[DatasetJob, ...] = (
    DatasetJob(
        name="bonedat",
        dataset_id=124,
        dataset_class=custom_dataset_io.BoneDat,
        source_root=Path("/mnt/nas/BoneHub/Public_Datasets/0124 BoneDat/BoneDat"),
    ),
    DatasetJob(
        name="kits2023",
        dataset_id=73,
        dataset_class=custom_dataset_io.KiTS2023,
        source_root=Path("/mnt/nas/BoneHub/Public_Datasets/0073 kits23"),
    ),
    DatasetJob(
        name="spine_mets_ct_seg",
        dataset_id=19,
        dataset_class=custom_dataset_io.SpineMetsCTSeg,
        source_root=Path("/mnt/nas/BoneHub/Public_Datasets/0019 TCIA Spine-Mets-CT-SEG"),
    ),
    DatasetJob(
        name="enhance_pet",
        dataset_id=134,
        dataset_class=custom_dataset_io.EnhancePET,
        source_root=Path("/mnt/nas/BoneHub/Public_Datasets/0134 enhance-pet-1_6k"),
    ),
    DatasetJob(
        name="vsd_lower_extremities_seg",
        dataset_id=180,
        dataset_class=custom_dataset_io.VSDLowerExtremitiesSeg,
        source_root=Path("/mnt/nas/BoneHub/Public_Datasets/0180 vsd-lower-extremities-seg"),
    ),
    DatasetJob(
        name="totalsegmentator_ct",
        dataset_id=27,
        dataset_class=custom_dataset_io.TotalSegmentatorCT,
        source_root=Path("/mnt/nas/BoneHub/Public_Datasets/0027 Totalsegmentator/v3.0.0"),
    ),
    DatasetJob(
        name="ctpelvic1k",
        dataset_id=44,
        dataset_class=custom_dataset_io.CTPelvic1K,
        source_root=Path("/mnt/nas/BoneHub/Public_Datasets/0044 CTPelvic1K"),
    ),
    DatasetJob(
        name="pengwin",
        dataset_id=106,
        dataset_class=custom_dataset_io.PENGWIN,
        source_root=Path("/mnt/nas/BoneHub/Public_Datasets/0106 PENGWIN"),
    ),
    DatasetJob(
        name="ctpel",
        dataset_id=138,
        dataset_class=custom_dataset_io.CTPEL,
        source_root=Path("/mnt/nas/BoneHub/Public_Datasets/0138 ctpel"),
    ),
    DatasetJob(
        name="synthrad2023",
        dataset_id=6,
        dataset_class=custom_dataset_io.SynthRAD2023,
        source_root=Path("/mnt/nas/BoneHub/Public_Datasets/0006 SynthRAD2023"),
    ),
    DatasetJob(
        name="vsd_full_body",
        dataset_id=71,
        dataset_class=custom_dataset_io.VSDFullBody,
        source_root=Path("/mnt/nas/BoneHub/Public_Datasets/0071 VSDFullBody"),
    ),
    DatasetJob(
        name="visible_human_3d_models",
        dataset_id=153,
        dataset_class=custom_dataset_io.VisibleHuman3DModels,
        source_root=Path("/mnt/nas/BoneHub/Public_Datasets/0153 visible-human-3d-models"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BoneHub public datasets in parallel.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root folder of the database the converted datasets are written into.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=len(DATASET_JOBS),
        help="Maximum number of datasets converted at once, each in its own process.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[job.name for job in DATASET_JOBS],
        help="Optional subset of dataset names to convert.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing dataset folders and convert them again.",
    )
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Keep the subjects an earlier run finished and convert the rest (e.g. those that failed).",
    )
    parser.add_argument(
        "--subject-workers",
        type=int,
        default=None,
        help="Subjects converted at once within each dataset, each in its own process "
        "(default: one per CPU core, at most 8, or 4 for vsd_full_body). Up to max-workers x subject-workers "
        "processes run together, each needing up to ~2 GB for the largest masks (~5 GB for vsd_full_body, ~6 GB for "
        "visible_human_3d_models); "
        "lower it if memory runs out.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide the per-dataset progress bars.",
    )
    return parser.parse_args()


def select_jobs(selected_names: Sequence[str] | None) -> list[DatasetJob]:
    if not selected_names:
        return list(DATASET_JOBS)
    selected = set(selected_names)
    return [job for job in DATASET_JOBS if job.name in selected]


def convert_dataset(
    job: DatasetJob,
    output_root: Path,
    overwrite: bool,
    resume: bool,
    verbose: bool,
    subject_workers: int | None = None,
) -> Path:
    dataset = job.dataset_class(job.source_root)
    return dataset.export_to_bonehub_format(
        output_root=output_root,
        output_dataset_id=job.dataset_id,
        overwrite=overwrite,
        resume=resume,
        num_workers=subject_workers,
        verbose=verbose,
    )


def setup_run_logger(output_root: Path) -> logging.Logger:
    """Log to the console and to a run log in <output_root>/logs, next to the logs of the datasets."""
    log_dir = output_root / layout.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger = logging.getLogger("convert_all_datasets")
    logger.setLevel(logging.INFO)
    logger.handlers = [logging.StreamHandler(), file_handler]
    logger.info(f"Run log: {log_file_path}")
    return logger


def log_summary(
    logger: logging.Logger, successes: list[tuple[DatasetJob, Path]], failures: list[tuple[DatasetJob, Exception]]
) -> None:
    lines = [f"Conversion summary: {len(successes)} succeeded, {len(failures)} failed."]
    for job, log_file_path in sorted(successes, key=lambda item: item[0].dataset_id):
        lines.append(f"  [OK]     {job.name} ({layout.dataset_dir_name(job.dataset_id)}), log: {log_file_path}")
    for job, exc in sorted(failures, key=lambda item: item[0].dataset_id):
        lines.append(f"  [FAILED] {job.name} ({layout.dataset_dir_name(job.dataset_id)}): {exc}")
    if failures:
        lines.append(
            "Refer to each failed dataset's log file to see which subjects failed and why. The subjects that failed "
            "have no folder; the other subjects were converted and saved. Run again with --resume to convert only "
            "the missing subjects."
        )
    logger.log(logging.ERROR if failures else logging.INFO, "\n".join(lines))


def main() -> int:
    args = parse_args()
    jobs = select_jobs(args.datasets)

    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")
    if not jobs:
        raise ValueError("No datasets selected for conversion.")

    output_root = args.output_root.resolve()
    logger = setup_run_logger(output_root)
    logger.info(f"Output root: {output_root}")
    logger.info(f"Datasets queued: {', '.join(job.name for job in jobs)}")
    logger.info(f"Converting up to {min(args.max_workers, len(jobs))} datasets at once")

    successes: list[tuple[DatasetJob, Path]] = []
    failures: list[tuple[DatasetJob, Exception]] = []

    # One process per dataset, so a crash in one does not stop the others.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(jobs)), mp_context=context) as executor:
        future_to_job = {
            executor.submit(
                convert_dataset,
                job,
                output_root,
                args.overwrite,
                args.resume,
                not args.quiet,
                args.subject_workers,
            ): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                successes.append((job, future.result()))
                logger.info(f"[OK] {job.name} -> {layout.dataset_dir_name(job.dataset_id)}")
            except Exception as exc:
                failures.append((job, exc))
                logger.error(f"[FAILED] {job.name} -> {layout.dataset_dir_name(job.dataset_id)}: {exc}")

    log_summary(logger, successes, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
