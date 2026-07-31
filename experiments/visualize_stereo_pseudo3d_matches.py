"""Create visual comparisons of stereo matches for pseudo-height profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_stereo_pseudo3d_density import (  # noqa: E402
    PROFILES,
    calibration,
    filtered_matches,
    gradient_features,
    select_mask,
    structures,
)


SELECTED = ("SIFT", "SIFT-dense", "Pseudo3D-dense", "Ours-selective")


def panel(keyframe: Path, profile_name: str, max_matches: int) -> np.ndarray:
    profile = next(item for item in PROFILES if item.name == profile_name)
    left_color = cv2.imread(str(keyframe / "Left_Image.png"), cv2.IMREAD_COLOR)
    right_color = cv2.imread(str(keyframe / "Right_Image.png"), cv2.IMREAD_COLOR)
    left = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=4000)
    left_kp, left_desc = sift.detectAndCompute(left, None)
    right_kp, right_desc = sift.detectAndCompute(right, None)
    left_h, left_angle, left_mask = gradient_features(left)
    right_h, right_angle, right_mask = gradient_features(right)
    left_kp, left_desc = select_mask(left_kp, left_desc, left_mask, profile.gradient_mask)
    right_kp, right_desc = select_mask(right_kp, right_desc, right_mask, profile.gradient_mask)
    matches = filtered_matches(
        profile,
        left_desc,
        right_desc,
        structures(left_kp, left_h, left_angle),
        structures(right_kp, right_h, right_angle),
    )
    if len(matches) > max_matches:
        indices = np.linspace(0, len(matches) - 1, max_matches, dtype=int)
        displayed = [matches[index] for index in indices]
    else:
        displayed = matches
    visualization = cv2.drawMatches(
        left_color,
        left_kp,
        right_color,
        right_kp,
        displayed,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    target_width = 1200
    scale = target_width / visualization.shape[1]
    visualization = cv2.resize(
        visualization,
        (target_width, round(visualization.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    title = f"{profile.name}: {len(matches)} accepted matches"
    cv2.rectangle(visualization, (0, 0), (target_width, 42), (255, 255, 255), -1)
    cv2.putText(
        visualization,
        title,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return visualization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-matches", type=int, default=120)
    args = parser.parse_args()
    panels = [panel(args.keyframe, name, args.max_matches) for name in SELECTED]
    separator = np.full((12, panels[0].shape[1], 3), 255, dtype=np.uint8)
    composite = np.vstack([item for panel_image in panels for item in (panel_image, separator)][:-1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), composite)


if __name__ == "__main__":
    main()
