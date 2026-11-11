"""Finding scans that are the same image, within and across datasets.

Every slice of an image, along each of its three array axes, gets a fingerprint: a hash of the histogram of its
values after subtracting the slice's minimum. A histogram does not change when a slice is rotated, flipped or
transposed, nor does subtracting the minimum change with a constant offset of the values. Fingerprints therefore
match whatever the file format, pixel type, header orientation (even a wrong one) or intensity offset, and slicing
along all three axes covers images whose axes were permuted. Slices with little information (air, padding) are
skipped, because they would match across unrelated images.

Two scans are the same image when at least `min_slices` slices match in an order that a crop, flip or shift
explains: their positions differ by a constant, or add up to one. Real CT and MR noise makes chance matches of whole
slices practically impossible, so such a match is confirmed without review. It finds re-saved, reoriented and cropped
copies; resampled or intensity-clipped copies have no identical slices and are not found.

Requires numpy and SimpleITK: pip install "bonehub-format[converter]".
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from . import layout
from .links import Link, load_links, merge_links, save_links
from .reader import Root

FINGERPRINT_VERSION = 1
#: A slice is informative if at least this share of its voxels lies above its minimum and it has this many values.
MIN_FOREGROUND = 0.05
MIN_DISTINCT_VALUES = 32
#: A fingerprint shared by more slices than this is a pattern, not a match (e.g. a constant phantom slice).
MAX_SHARED = 32


def _integer_slice(slice_: np.ndarray) -> np.ndarray:
    if np.issubdtype(slice_.dtype, np.floating):
        slice_ = np.rint(np.nan_to_num(slice_))
    return slice_.astype(np.int64, copy=False)


def slice_fingerprint(slice_: np.ndarray) -> int:
    """The fingerprint of one slice (see the module docstring), or 0 if the slice has too little information."""
    values = _integer_slice(slice_).ravel()
    values = values - values.min()
    if np.count_nonzero(values) < MIN_FOREGROUND * values.size:
        return 0
    histogram = np.bincount(values)
    if np.count_nonzero(histogram) < MIN_DISTINCT_VALUES:
        return 0
    digest = hashlib.blake2b(histogram.astype("<i8").tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "little") or 1


def image_fingerprints(image_path: Path) -> list[np.ndarray]:
    """For each array axis of an image, the fingerprint of every slice along it (0: not informative)."""
    image = sitk.ReadImage(str(image_path))  # kept in a variable: the array is a view of its memory
    array = sitk.GetArrayViewFromImage(image)
    if array.ndim != 3:
        raise ValueError(f"'{image_path}' is not a 3D image.")
    return [
        np.array([slice_fingerprint(np.take(array, i, axis=axis)) for i in range(array.shape[axis])], dtype=np.uint64)
        for axis in range(3)
    ]


def _cache_path(root: Path, key: str) -> Path:
    return root / layout.CACHE_DIR / "fingerprints" / f"{key.replace('/', '_')}.npz"


def cached_fingerprints(root: Path, key: str, image_path: Path) -> list[np.ndarray]:
    """The fingerprints of a scan's image, kept in <root>/.cache until the image changes."""
    stat = image_path.stat()
    stamp = np.array([FINGERPRINT_VERSION, stat.st_size, stat.st_mtime_ns], dtype=np.int64)
    cache = _cache_path(root, key)
    if cache.is_file():
        with np.load(cache) as stored:
            if np.array_equal(stored["stamp"], stamp):
                return [stored[f"axis{axis}"] for axis in range(3)]
    fingerprints = image_fingerprints(image_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, stamp=stamp, **{f"axis{axis}": fingerprints[axis] for axis in range(3)})
    return fingerprints


def _fingerprint_job(job: tuple[Path, str, Path]) -> tuple[str, list[np.ndarray] | None, str | None]:
    root, key, image_path = job
    try:
        return key, cached_fingerprints(root, key, image_path), None
    except Exception as exc:  # reported, not fatal: one unreadable image should not stop the search
        return key, None, repr(exc)


def compute_fingerprints(root: Path, workers: int | None = None, verbose: bool = True) -> tuple[dict, dict]:
    """The fingerprints of every scan image of a root, and the errors of the images that could not be read."""
    root = Path(root)
    jobs = [(root, scan.key, scan.image_path) for scan in Root(root).scans() if scan.image_path]
    workers = workers or min(os.cpu_count() or 4, 4)
    fingerprints, errors = {}, {}

    def collect(results):
        for key, value, error in tqdm(results, total=len(jobs), desc="Fingerprinting images", unit="scan", disable=not verbose):
            if error:
                errors[key] = error
            else:
                fingerprints[key] = value

    if workers == 1:
        collect(map(_fingerprint_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            collect(pool.map(_fingerprint_job, jobs, chunksize=1))
    return fingerprints, errors


def match_fingerprints(fingerprints: dict[str, list[np.ndarray]], min_slices: int = 5) -> list[Link]:
    """same_image links between scans whose fingerprints match (see the module docstring)."""
    where = defaultdict(list)  # fingerprint -> (scan, axis, slice index)
    for key, axes in fingerprints.items():
        for axis, values in enumerate(axes):
            for index in np.flatnonzero(values):
                where[int(values[index])].append((key, axis, int(index)))

    pairs = defaultdict(list)  # (scan a, axis a, scan b, axis b) -> [(index a, index b)]
    for places in where.values():
        if len(places) > MAX_SHARED or len({key for key, _, _ in places}) < 2:
            continue
        for i, (key_a, axis_a, index_a) in enumerate(places):
            for key_b, axis_b, index_b in places[i + 1 :]:
                if key_a == key_b:
                    continue
                if key_a > key_b:
                    key_a, axis_a, index_a, key_b, axis_b, index_b = key_b, axis_b, index_b, key_a, axis_a, index_a
                pairs[key_a, axis_a, key_b, axis_b].append((index_a, index_b))

    best: dict[tuple[str, str], int] = {}
    for (key_a, _, key_b, _), matches in pairs.items():
        shifts = Counter(b - a for a, b in matches).most_common(1)[0][1]
        flips = Counter(b + a for a, b in matches).most_common(1)[0][1]
        score = max(shifts, flips)
        if score >= min_slices and score > best.get((key_a, key_b), 0):
            best[key_a, key_b] = score
    return [
        Link(scans=pair, relation="same_image", evidence="identical_slices", score=score, decision="confirmed")
        for pair, score in sorted(best.items())
    ]


def find_identical_images(
    root: Path, min_slices: int = 5, workers: int | None = None, update: bool = True, verbose: bool = True
) -> tuple[list[Link], dict[str, str]]:
    """Find the scans of a root that are the same image, and merge them into links.json (unless update is False).

    Returns the links found and, per scan whose image could not be read, the error.
    """
    root = Path(root)
    fingerprints, errors = compute_fingerprints(root, workers, verbose)
    found = match_fingerprints(fingerprints, min_slices)
    if update:
        save_links(root, merge_links(load_links(root), found))
    return found, errors
