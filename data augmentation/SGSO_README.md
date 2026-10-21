# SGSO: Skeleton-Guided Smart Occlusion

This directory contains the public implementation of **Skeleton-Guided Smart Occlusion (SGSO)** used in our gait-recognition experiments on **CSCZED** and **SUSTech1K**.

SGSO is a training-time data augmentation strategy designed to synthesize clothing-like occlusions from non-occluded gait sequences. The same SGSO core implementation is shared by both datasets, while dataset-specific data loading, sequence selection, and preprocessing are handled separately.

## File Structure

```text
SGSO/
├── sgso.py
├── augment_csczed.py
├── augment_sustech1k.py
└── README.md
```

- `sgso.py`: shared SGSO implementation and spatial normalization utilities.
- `augment_csczed.py`: CSCZED-specific sequence selection, loading, augmentation, and saving.
- `augment_sustech1k.py`: SUSTech1K-specific loading, spatial normalization, augmentation, and saving.

## Dependencies

```bash
pip install numpy opencv-python tqdm
```

Python 3.9 or later is recommended.

## Method Overview

For each selected sequence, SGSO constructs a clothing-like occlusion region from 2D skeleton keypoints.

The main steps are:

1. Locate the left and right shoulders and the available knee keypoints.
2. Compute the shoulder reference height.
3. Estimate the vertical body extent between the shoulder line and the knee.
4. Sample an occlusion-length ratio.
5. Construct a trapezoidal occlusion mask.
6. Expand the lower boundary horizontally by a sampled pixel offset.
7. Apply the same geometric mask to the silhouette and parsing representations.
8. Keep the sampled augmentation parameters fixed across all frames of the same sequence.

This sequence-level parameter sharing preserves temporal consistency.

## Spatial Normalization

Before SGSO, the spatial scale is normalized according to the foreground height.

Given a silhouette foreground with vertical extent

```text
H = y_max - y_min + 1
```

the scale factor is

```text
scale = target_foreground_height / H
```

The default target foreground height is:

```text
240 pixels
```

The complete image canvas is resized using this scale factor rather than cropping the foreground to a fixed image size.

For example:

```text
original image height     = 280
foreground height         = 200
target foreground height  = 240

scale = 240 / 200 = 1.2
new image height = 280 × 1.2 = 336
```

The image width is scaled by the same factor.

The silhouette, parsing map, and skeleton coordinates are all transformed using the same scale factor:

```text
x' = x × scale
y' = y × scale
```

This keeps the modalities spatially aligned.

Nearest-neighbor interpolation is used for silhouette and parsing maps because they contain discrete labels.

## SGSO Parameters

The default augmentation parameters are:

```text
occlusion label        = 11
occlusion ratio range  = [0.9, 1.2]
horizontal expansion   = [6, 10] pixels
upper expansion        = 4 pixels
foreground height      = 240 pixels
```

The occlusion ratio and horizontal expansion are sampled once per sequence and remain fixed across all frames.

The scripts support changing these values through command-line arguments.

## Parsing Update

The synthetic clothing region is first defined geometrically.

For the parsing representation, selected semantic labels are protected from replacement, while the remaining pixels inside the synthetic clothing region are reassigned to the clothing label.

The released implementation also preserves the label-replacement behavior used in the experimental code.

## CSCZED

### Dataset-Specific Sampling

For CSCZED, SGSO is applied to selected non-clothing source sequences.

For each eligible subject:

1. `LCL` and `MCL` source types are excluded from the candidate pool.
2. Three remaining activity types are randomly selected.
3. One trajectory is randomly selected for each selected activity type.
4. All available views of the selected trajectories are augmented.
5. One SGSO parameter pair is sampled for each sequence and kept fixed across its frames.

### Expected CSCZED Input Structure

```text
pose_root/
  <subject>/<type>/<view>/<view>.pkl

silhouette_root/
  <subject>/<type>/<view>/<view>.pkl

parsing_root/
  <subject>/<type>/<view>/<view>.pkl

subject_root/
  <subject>/
```

Each `.pkl` file contains the corresponding sequence.

### Example

```bash
python augment_csczed.py \
  --pose-root /path/to/pose_map \
  --silhouette-root /path/to/silhouettes \
  --parsing-root /path/to/parsing \
  --subject-root /path/to/eligible_subjects \
  --output-silhouette-root /path/to/output/silhouettes \
  --output-parsing-root /path/to/output/parsing
```

## SUSTech1K

### Dataset-Specific Processing

For SUSTech1K:

1. the pose sequence is loaded from the released pose file;
2. silhouette frames are loaded from `Sils_aligned`;
3. parsing frames are loaded from the corresponding parsing directory;
4. the silhouette, parsing map, and skeleton coordinates are normalized using the same foreground-height scale;
5. one SGSO parameter pair is sampled for each sequence;
6. the same parameters are applied to all frames;
7. augmented silhouette and parsing sequences are saved as `.pkl` files.

### Expected SUSTech1K Input Structure

```text
pose_root/
  <subject>/<type>/<view>/03-<view>-Camera-Pose.pkl

silhouette_root/
  <subject>/<type>/<view>/Sils_aligned/<frame files>

parsing_root/
  <subject>/<type>/<view>/<frame files>

reference_root/
  <subject>/<type>/<view>/
```

`reference_root` defines the sequences to be processed.

### Example

```bash
python augment_sustech1k.py \
  --pose-root /path/to/SUSTech1K-Released-pkl \
  --silhouette-root /path/to/SUSTech1K-Released-2023 \
  --parsing-root /path/to/parsing \
  --reference-root /path/to/selected_sequences \
  --output-silhouette-root /path/to/output/silhouettes \
  --output-parsing-root /path/to/output/parsing
```

## Reproducibility

Both dataset-specific scripts support deterministic random sampling:

```bash
--seed 0
```

A deterministic random generator is derived from the global seed and the subject or sequence identifier. This makes subject-level sequence selection, SGSO parameter sampling, and multi-threaded preprocessing reproducible.

The sampled SGSO parameters remain fixed across all frames of one sequence.

## Output

For both datasets, the augmented outputs are saved as sequence-level `.pkl` files.

```text
output_silhouette_root/
  <subject>/<type>/<view>/<view>.pkl

output_parsing_root/
  <subject>/<type>/<view>/<view>.pkl
```

## Notes

- SGSO is used as a training-time augmentation strategy.
- CSCZED and SUSTech1K share the same `skeleton_guided_smart_occlusion` implementation.
- Dataset-specific sampling and preprocessing remain separate.
- SGSO does not modify the recognition network itself.
- The spatial normalization step uses foreground height rather than a fixed whole-image resize, while preserving cross-modal alignment.
