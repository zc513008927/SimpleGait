"""
SGSO preprocessing for CSCZED.

Dataset-specific logic:
- load pose, silhouette and parsing sequences from .pkl files;
- exclude LCL/MCL from candidate source types;
- randomly select three eligible activity types per subject;
- randomly select one trajectory from each selected activity type;
- augment all available views of each selected trajectory;
- sample one SGSO parameter pair per sequence and keep it fixed across frames.

The shared SGSO implementation is identical to the SUSTech1K version.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Tuple

from tqdm import tqdm

from sgso import resize_by_foreground_height, skeleton_guided_smart_occlusion


SequenceInfo = Tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply SGSO to CSCZED sequences."
    )

    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--silhouette-root", type=Path, required=True)
    parser.add_argument("--parsing-root", type=Path, required=True)

    parser.add_argument(
        "--subject-root",
        type=Path,
        required=True,
        help="Directory whose subject folders define eligible CSCZED subjects.",
    )

    parser.add_argument("--output-silhouette-root", type=Path, required=True)
    parser.add_argument("--output-parsing-root", type=Path, required=True)

    parser.add_argument("--occlusion-label", type=int, default=11)
    parser.add_argument("--ratio-min", type=float, default=0.9)
    parser.add_argument("--ratio-max", type=float, default=1.2)
    parser.add_argument("--ratio-decimals", type=int, default=1)
    parser.add_argument("--expansion-min", type=int, default=6)
    parser.add_argument("--expansion-max", type=int, default=10)
    parser.add_argument(
        "--target-foreground-height",
        type=int,
        default=240,
        help="Normalize silhouette foreground height to this value.",
    )

    parser.add_argument(
        "--types-per-subject",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-subject-id",
        type=int,
        default=81,
        help="Preserves the subject-ID cutoff in the supplied CSCZED script.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 1) + 4),
    )

    return parser.parse_args()


def make_rng(seed: int, key: str) -> random.Random:
    """Create a deterministic RNG for subject/sequence-level sampling."""
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    derived_seed = int.from_bytes(digest[:8], "little", signed=False)
    return random.Random(derived_seed)


def split_type_name(name: str):
    """Split ACT-TRAJ into activity and trajectory."""
    if "-" not in name:
        return None

    activity, trajectory = name.split("-", 1)
    return activity, trajectory


def collect_tasks(args: argparse.Namespace) -> List[SequenceInfo]:
    """
    Collect CSCZED tasks using the original dataset-specific sampling rule.

    Note:
        The supplied research script appended [_id, _type, _view] after
        selecting _t. Here this is corrected to [_id, _t, _view].
    """
    tasks: List[SequenceInfo] = []

    subject_ids = sorted(
        path.name
        for path in args.subject_root.iterdir()
        if path.is_dir()
    )

    for subject_id in subject_ids:
        if args.max_subject_id >= 0:
            try:
                if int(subject_id) > args.max_subject_id:
                    continue
            except ValueError:
                pass

        subject_pose_dir = args.pose_root / subject_id
        if not subject_pose_dir.is_dir():
            continue

        trajectories_by_activity: DefaultDict[str, List[str]] = defaultdict(list)

        for type_dir in sorted(
            path for path in subject_pose_dir.iterdir() if path.is_dir()
        ):
            parsed = split_type_name(type_dir.name)
            if parsed is None:
                continue

            activity, trajectory = parsed
            activity_upper = activity.upper()

            if "LCL" in activity_upper or "MCL" in activity_upper:
                continue

            trajectories_by_activity[activity].append(trajectory)

        activities = sorted(trajectories_by_activity.keys())

        if len(activities) < args.types_per_subject:
            tqdm.write(
                f"Skip {subject_id}: only {len(activities)} eligible type(s)."
            )
            continue

        subject_rng = make_rng(args.seed, f"subject:{subject_id}")

        selected_activities = subject_rng.sample(
            activities,
            args.types_per_subject,
        )

        selected_types = []

        for activity in selected_activities:
            trajectories = sorted(
                trajectories_by_activity[activity]
            )

            selected_trajectory = subject_rng.choice(trajectories)
            selected_types.append(
                f"{activity}-{selected_trajectory}"
            )

        for sequence_type in selected_types:
            view_root = subject_pose_dir / sequence_type

            if not view_root.is_dir():
                continue

            for view_dir in sorted(
                path for path in view_root.iterdir() if path.is_dir()
            ):
                # Corrected from the original _type to the selected _t.
                tasks.append(
                    (subject_id, sequence_type, view_dir.name)
                )

    return tasks


def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


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
        / f"{view}.pkl"
    )

    silhouette_path = (
        args.silhouette_root
        / subject_id
        / sequence_type
        / view
        / f"{view}.pkl"
    )

    parsing_path = (
        args.parsing_root
        / subject_id
        / sequence_type
        / view
        / f"{view}.pkl"
    )

    pose_sequence = load_pickle(pose_path)
    silhouette_sequence = load_pickle(silhouette_path)
    parsing_sequence = load_pickle(parsing_path)

    if not (
        len(pose_sequence)
        == len(silhouette_sequence)
        == len(parsing_sequence)
    ):
        raise ValueError(
            f"Frame-count mismatch for {'/'.join(sequence)}: "
            f"pose={len(pose_sequence)}, "
            f"silhouette={len(silhouette_sequence)}, "
            f"parsing={len(parsing_sequence)}"
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

    for pose, silhouette, parsing in zip(
        pose_sequence,
        silhouette_sequence,
        parsing_sequence,
    ):
        # Normalize foreground height to 240 pixels. The full canvas,
        # parsing map, and skeleton coordinates are scaled together.
        silhouette_norm, parsing_norm, pose_norm = resize_by_foreground_height(
            silhouette_map=silhouette,
            parsing_map=parsing,
            keypoints=pose,
            target_foreground_height=args.target_foreground_height,
        )

        parsing_aug, silhouette_aug = skeleton_guided_smart_occlusion(
            parsing_map=parsing_norm,
            silhouette_map=silhouette_norm,
            keypoints=pose_norm,
            occlusion_label=args.occlusion_label,
            expansion_pixels=expansion_pixels,
            augmentation_prob=0.5,
            occlusion_length_ratio=occlusion_length_ratio,
            expansion_pixels_upper=4,
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
    tasks = collect_tasks(args)

    if not tasks:
        raise RuntimeError("No CSCZED sequences were selected.")

    print(f"Selected {len(tasks)} sequences.")
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
            desc="SGSO-CSCZED",
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

    print("CSCZED SGSO preprocessing completed.")


if __name__ == "__main__":
    main()
