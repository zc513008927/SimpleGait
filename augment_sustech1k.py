"""
SGSO preprocessing for SUSTech1K.

Dataset-specific logic:
- read raw silhouette/parsing frame images;
- align them to 240x120 before augmentation;
- sample one SGSO parameter pair per sequence;
- keep the sampled parameters fixed across all frames of that sequence;
- process sequences listed under reference_root.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import pickle
import random
from pathlib import Path
from typing import List, Tuple

import cv2
from tqdm import tqdm

from sgso import resize_by_foreground_height, skeleton_guided_smart_occlusion


SequenceInfo = Tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply SGSO to SUSTech1K sequences."
    )

    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--silhouette-root", type=Path, required=True)
    parser.add_argument("--parsing-root", type=Path, required=True)

    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
        help="Directory tree defining the sequences to process.",
    )

    parser.add_argument("--output-silhouette-root", type=Path, required=True)
    parser.add_argument("--output-parsing-root", type=Path, required=True)

    parser.add_argument("--occlusion-label", type=int, default=11)
    parser.add_argument("--ratio-min", type=float, default=0.9)
    parser.add_argument("--ratio-max", type=float, default=1.2)
    parser.add_argument("--ratio-decimals", type=int, default=1)
    parser.add_argument("--expansion-min", type=int, default=6)
    parser.add_argument("--expansion-max", type=int, default=10)
    parser.add_argument("--upper-expansion", type=int, default=4)

    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--width",
        type=int,
        default=120,
        help="Reserved for compatibility; width is scaled proportionally.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 1) + 4),
    )

    return parser.parse_args()


def make_rng(seed: int, key: str) -> random.Random:
    """Create a deterministic RNG for one sequence."""
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    derived_seed = int.from_bytes(digest[:8], "little", signed=False)
    return random.Random(derived_seed)


def collect_tasks(reference_root: Path) -> List[SequenceInfo]:
    """Collect subject/type/view sequences from the reference directory."""
    tasks: List[SequenceInfo] = []

    for subject_dir in sorted(
        path for path in reference_root.iterdir() if path.is_dir()
    ):
        for type_dir in sorted(
            path for path in subject_dir.iterdir() if path.is_dir()
        ):
            for view_dir in sorted(
                path for path in type_dir.iterdir() if path.is_dir()
            ):
                tasks.append(
                    (subject_dir.name, type_dir.name, view_dir.name)
                )

    return tasks


def read_gray(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def process_sequence(
    sequence: SequenceInfo,
    args: argparse.Namespace,
) -> str:
    subject_id, sequence_type, view = sequence

    pose_path = (
        args.pose_root
        / subject_id
        / sequence_type
        / view
        / f"03-{view}-Camera-Pose.pkl"
    )

    silhouette_dir = (
        args.silhouette_root
        / subject_id
        / sequence_type
        / view
        / "Sils_aligned"
    )

    parsing_dir = (
        args.parsing_root
        / subject_id
        / sequence_type
        / view
    )

    with pose_path.open("rb") as file:
        pose_sequence = pickle.load(file)

    silhouette_files = sorted(
        path for path in silhouette_dir.iterdir() if path.is_file()
    )
    parsing_files = sorted(
        path for path in parsing_dir.iterdir() if path.is_file()
    )

    if not (
        len(pose_sequence)
        == len(silhouette_files)
        == len(parsing_files)
    ):
        raise ValueError(
            f"Frame-count mismatch for {'/'.join(sequence)}: "
            f"pose={len(pose_sequence)}, "
            f"silhouette={len(silhouette_files)}, "
            f"parsing={len(parsing_files)}"
        )

    rng = make_rng(args.seed, "/".join(sequence))

    occlusion_length_ratio = round(
        rng.uniform(args.ratio_min, args.ratio_max),
        args.ratio_decimals,
    )
    expansion_pixels = rng.randint(
        args.expansion_min,
        args.expansion_max,
    )

    augmented_parsing = []
    augmented_silhouettes = []

    for pose, silhouette_file, parsing_file in zip(
        pose_sequence,
        silhouette_files,
        parsing_files,
    ):
        silhouette = read_gray(silhouette_file)
        parsing = read_gray(parsing_file)

        # Normalize the foreground height to 240 pixels while scaling the
        # complete image canvas, parsing map, and skeleton coordinates by the
        # same factor. This keeps all modalities spatially aligned.
        silhouette_norm, parsing_norm, aligned_pose = resize_by_foreground_height(
            silhouette_map=silhouette,
            parsing_map=parsing,
            keypoints=pose,
            target_foreground_height=args.height,
        )

        parsing_aug, silhouette_aug = skeleton_guided_smart_occlusion(
            parsing_map=parsing_norm,
            silhouette_map=silhouette_norm,
            keypoints=aligned_pose,
            occlusion_label=args.occlusion_label,
            expansion_pixels=expansion_pixels,
            augmentation_prob=0.5,
            occlusion_length_ratio=occlusion_length_ratio,
            expansion_pixels_upper=args.upper_expansion,
        )

        augmented_parsing.append(parsing_aug)
        augmented_silhouettes.append(silhouette_aug)

    silhouette_output = (
        args.output_silhouette_root
        / subject_id
        / sequence_type
        / view
    )
    parsing_output = (
        args.output_parsing_root
        / subject_id
        / sequence_type
        / view
    )

    silhouette_output.mkdir(parents=True, exist_ok=True)
    parsing_output.mkdir(parents=True, exist_ok=True)

    with (silhouette_output / f"{view}.pkl").open("wb") as file:
        pickle.dump(augmented_silhouettes, file)

    with (parsing_output / f"{view}.pkl").open("wb") as file:
        pickle.dump(augmented_parsing, file)

    return (
        f"OK {'/'.join(sequence)} "
        f"(ratio={occlusion_length_ratio}, expansion={expansion_pixels})"
    )


def main() -> None:
    args = parse_args()
    tasks = collect_tasks(args.reference_root)

    if not tasks:
        raise RuntimeError("No SUSTech1K sequences found.")

    print(f"Found {len(tasks)} sequences.")
    print(f"Using {args.workers} workers.")

    failures = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        future_to_task = {
            executor.submit(process_sequence, task, args): task
            for task in tasks
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=len(future_to_task),
            desc="SGSO-SUSTech1K",
        ):
            task = future_to_task[future]

            try:
                future.result()
            except Exception as exc:
                failures.append((task, exc))
                tqdm.write(f"ERROR {'/'.join(task)}: {exc}")

    if failures:
        raise RuntimeError(
            f"Finished with {len(failures)} failed sequence(s)."
        )

    print("SUSTech1K SGSO preprocessing completed.")


if __name__ == "__main__":
    main()
