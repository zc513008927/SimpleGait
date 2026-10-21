"""
Shared Skeleton-Guided Smart Occlusion (SGSO) implementation.

This module is shared by the CSCZED and SUSTech1K preprocessing scripts.
The dataset-specific sequence selection and I/O logic are kept in separate files.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


KEYPOINT_INDEX = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
}

# Parsing labels protected from geometric replacement.
PROTECTED_LABELS = (3, 4, 5, 6)

# Preserve the behavior of the original implementation:
# parsing label 2 is reassigned to the synthetic clothing label.
FORCE_REPLACE_LABELS = (2,)


def get_keypoint(
    keypoints: np.ndarray,
    index: int,
) -> Optional[Tuple[int, int]]:
    """Return one keypoint as (x, y), or None if it is unavailable."""
    if index >= len(keypoints):
        return None

    point = np.asarray(keypoints[index])
    if point.size < 2:
        return None

    if point.size >= 3 and point[2] < 0:
        return None

    return int(round(float(point[0]))), int(round(float(point[1])))


def skeleton_guided_smart_occlusion(
    parsing_map: np.ndarray,
    silhouette_map: np.ndarray,
    keypoints: np.ndarray,
    occlusion_label: int = 11,
    expansion_pixels: int = 8,
    augmentation_prob: float = 0.5,
    occlusion_length_ratio: float = 1.0,
    expansion_pixels_upper: int = 4,
    protected_labels: Sequence[int] = PROTECTED_LABELS,
    force_replace_labels: Sequence[int] = FORCE_REPLACE_LABELS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Skeleton-Guided Smart Occlusion.

    The geometric mask is constructed from the shoulder line and knee position.
    The same mask is used to enlarge the silhouette and to update the parsing map.

    Args:
        parsing_map: Single-channel human parsing map.
        silhouette_map: Single-channel silhouette/GPS foreground map.
        keypoints: COCO-style keypoints containing at least x and y coordinates.
        occlusion_label: Parsing label used for the synthetic clothing region.
        expansion_pixels: Horizontal expansion of the lower boundary.
        augmentation_prob: Kept for interface compatibility with the research code.
                           Sequence selection is handled by the dataset-specific driver.
        occlusion_length_ratio: Vertical clothing length ratio.
        expansion_pixels_upper: Horizontal expansion at the shoulder boundary.
        protected_labels: Parsing labels excluded from geometric replacement.
        force_replace_labels: Parsing labels always reassigned to occlusion_label.

    Returns:
        augmented_parsing_map, augmented_silhouette_map
    """
    del augmentation_prob  # selection is handled outside this function

    points = {
        name: get_keypoint(keypoints, index)
        for name, index in KEYPOINT_INDEX.items()
    }

    shoulders_ok = (
        points["left_shoulder"] is not None
        and points["right_shoulder"] is not None
    )
    knee_ok = (
        points["left_knee"] is not None
        or points["right_knee"] is not None
    )

    if not shoulders_ok or not knee_ok:
        return parsing_map.copy(), silhouette_map.copy()

    h, w = silhouette_map.shape

    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]
    assert left_shoulder is not None
    assert right_shoulder is not None

    if left_shoulder[0] < right_shoulder[0]:
        top_left, top_right = left_shoulder, right_shoulder
    else:
        top_left, top_right = right_shoulder, left_shoulder

    shoulder_y = (top_left[1] + top_right[1]) // 2

    knee_ys = [
        point[1]
        for name, point in points.items()
        if "knee" in name and point is not None
    ]
    knee_y = max(knee_ys)

    body_height = knee_y - shoulder_y
    lower_y = shoulder_y + int(body_height * occlusion_length_ratio)
    lower_y = int(np.clip(lower_y, 0, h - 1))

    foreground_x = np.where(parsing_map[lower_y, :] > 0)[0]

    if foreground_x.size > 0:
        lower_left_x = int(foreground_x.min())
        lower_right_x = int(foreground_x.max())
    else:
        left_hip = points["left_hip"]
        right_hip = points["right_hip"]

        if left_hip is None or right_hip is None:
            return parsing_map.copy(), silhouette_map.copy()

        lower_left_x = left_hip[0]
        lower_right_x = right_hip[0]

    lower_left_x = max(0, lower_left_x - expansion_pixels)
    lower_right_x = min(w - 1, lower_right_x + expansion_pixels)

    polygon = np.asarray(
        [
            [top_left[0] - expansion_pixels_upper, top_left[1]],
            [top_right[0] + expansion_pixels_upper, top_right[1]],
            [lower_right_x, lower_y],
            [lower_left_x, lower_y],
        ],
        dtype=np.int32,
    )

    # Silhouette augmentation: S_aug = S union M.
    augmented_silhouette = silhouette_map.copy()
    cv2.fillPoly(augmented_silhouette, [polygon], 255)

    # Parsing augmentation.
    geometric_mask = np.zeros_like(parsing_map, dtype=np.uint8)
    cv2.fillPoly(geometric_mask, [polygon], 255)

    protected_mask = np.isin(parsing_map, protected_labels)
    effective_mask = (geometric_mask > 0) & (~protected_mask)

    if force_replace_labels:
        effective_mask |= np.isin(parsing_map, force_replace_labels)

    augmented_parsing = parsing_map.copy()
    augmented_parsing[effective_mask] = occlusion_label

    return augmented_parsing, augmented_silhouette



def resize_by_foreground_height(
    silhouette_map: np.ndarray,
    parsing_map: np.ndarray,
    keypoints: np.ndarray,
    target_foreground_height: int = 240,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resize all spatial inputs using the silhouette foreground height.

    The scale factor is computed from the vertical foreground extent of the
    silhouette:

        scale = target_foreground_height / foreground_height

    The *whole image canvas* is resized by this factor rather than cropping
    the foreground to 240 pixels. The parsing map and skeleton coordinates
    use exactly the same scale factor, preserving their spatial alignment.

    Example:
        If an input image is 280 pixels high and its foreground occupies
        200 pixels vertically, the scale factor is 240 / 200 = 1.2.
        The resized image height becomes 280 * 1.2 = 336 pixels, and the
        width and skeleton coordinates are scaled by the same factor.

    Args:
        silhouette_map: Single-channel silhouette/GPS foreground map.
        parsing_map: Single-channel parsing map aligned with the silhouette.
        keypoints: Keypoints containing x, y, and optionally confidence.
        target_foreground_height: Desired foreground height in pixels.

    Returns:
        resized_silhouette, resized_parsing, resized_keypoints
    """
    if silhouette_map.shape[:2] != parsing_map.shape[:2]:
        raise ValueError(
            "Silhouette and parsing maps must have the same spatial size "
            f"before resizing, got {silhouette_map.shape[:2]} and "
            f"{parsing_map.shape[:2]}."
        )

    foreground_y = np.where(silhouette_map > 0)[0]

    if foreground_y.size == 0:
        # No reliable foreground extent is available. Keep the inputs unchanged.
        return (
            silhouette_map.copy(),
            parsing_map.copy(),
            np.asarray(keypoints, dtype=np.float32).copy(),
        )

    y_min = int(foreground_y.min())
    y_max = int(foreground_y.max())
    foreground_height = y_max - y_min + 1

    if foreground_height <= 0:
        return (
            silhouette_map.copy(),
            parsing_map.copy(),
            np.asarray(keypoints, dtype=np.float32).copy(),
        )

    scale = float(target_foreground_height) / float(foreground_height)

    input_height, input_width = silhouette_map.shape[:2]
    output_height = max(1, int(round(input_height * scale)))
    output_width = max(1, int(round(input_width * scale)))

    resized_silhouette = cv2.resize(
        silhouette_map,
        (output_width, output_height),
        interpolation=cv2.INTER_NEAREST,
    )
    resized_parsing = cv2.resize(
        parsing_map,
        (output_width, output_height),
        interpolation=cv2.INTER_NEAREST,
    )

    resized_keypoints = np.asarray(keypoints, dtype=np.float32).copy()

    if resized_keypoints.ndim != 2 or resized_keypoints.shape[1] < 2:
        raise ValueError(
            "keypoints must have shape [N, >=2], containing x and y coordinates."
        )

    if resized_keypoints.shape[1] >= 3:
        valid = resized_keypoints[:, 2] >= 0
    else:
        valid = np.ones(len(resized_keypoints), dtype=bool)

    resized_keypoints[valid, 0] *= scale
    resized_keypoints[valid, 1] *= scale

    return resized_silhouette, resized_parsing, resized_keypoints


def align_by_centroid(
    mask: np.ndarray,
    keypoints: np.ndarray,
    output_height: int = 240,
    output_width: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize a mask and align its keypoints on a person-centered canvas.

    This function is used by the SUSTech1K preprocessing script, where raw
    silhouette/parsing frame images are aligned before SGSO.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32).copy()

    ys, xs = np.where(mask > 0)

    if xs.size == 0:
        x_min, y_min = 0, 0
        x_max = mask.shape[1] - 1
        y_max = mask.shape[0] - 1
    else:
        x_min, y_min = int(xs.min()), int(ys.min())
        x_max, y_max = int(xs.max()), int(ys.max())

    cropped = mask[y_min:y_max + 1, x_min:x_max + 1].copy()
    crop_h, crop_w = cropped.shape

    scale = min(output_height / crop_h, output_width / crop_w)
    resized_width = max(1, int(crop_w * scale))

    resized = cv2.resize(
        cropped,
        (resized_width, output_height),
        interpolation=cv2.INTER_NEAREST,
    )

    column_sum = resized.sum(axis=0)
    if column_sum.sum() > 0:
        center_x = int(
            np.searchsorted(np.cumsum(column_sum), column_sum.sum() / 2)
        )
    else:
        center_x = resized.shape[1] // 2

    offset_x = int(output_width / 2 - center_x)
    offset_x = max(0, min(offset_x, output_width - resized.shape[1]))

    canvas = np.zeros((output_height, output_width), dtype=np.uint8)
    canvas[:, offset_x:offset_x + resized.shape[1]] = resized

    valid = [
        (float(x), float(y))
        for x, y, confidence in keypoints
        if confidence >= 0
    ]

    if not valid:
        valid = [(0.0, 0.0)]

    xs_k, ys_k = zip(*valid)
    xmin_k, ymin_k = min(xs_k), min(ys_k)
    ymax_k = max(ys_k)

    keypoint_height = max(1.0, ymax_k - ymin_k + 1.0)
    keypoint_scale = (output_height / keypoint_height) * 0.9

    for idx, (x, y, confidence) in enumerate(keypoints):
        if confidence < 0:
            continue

        keypoints[idx, 0] = (x - xmin_k) * keypoint_scale
        keypoints[idx, 1] = (y - ymin_k) * keypoint_scale

    torso_indices = (11, 12, 5, 6)
    selected_x = [
        keypoints[idx, 0]
        for idx in torso_indices
        if idx < len(keypoints) and keypoints[idx, 2] >= 0
    ]

    if not selected_x:
        selected_x = [x for x, _ in valid]

    shift_x = output_width // 2 - float(np.mean(selected_x))

    valid_rows = keypoints[:, 2] >= 0
    keypoints[valid_rows, 0] += shift_x
    keypoints[valid_rows, 1] += 10

    # Preserve the original shoulder/elbow/wrist separation adjustment.
    for left_idx, right_idx in ((5, 6), (7, 8), (9, 10)):
        if left_idx >= len(keypoints) or right_idx >= len(keypoints):
            continue

        if keypoints[left_idx, 0] > keypoints[right_idx, 0]:
            keypoints[left_idx, 0] -= 4
            keypoints[right_idx, 0] += 4
        else:
            keypoints[left_idx, 0] += 4
            keypoints[right_idx, 0] -= 4

    return canvas, keypoints
