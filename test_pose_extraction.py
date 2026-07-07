"""Portable smoke check for the keypoint dataset layout.

The original script depended on a Windows-only checkout path and a removed
`ctr_gcn/prepare_pose_data.py` module. This version keeps the intent of a
sanity check without crashing on Linux or in a trimmed repository.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(
    sys.argv[1] if len(sys.argv) > 1 else PROJECT_ROOT / "DATASET_ug_sign_language"
)
FEATURES_DIR = DATASET_ROOT / "features"


def main() -> None:
    if not DATASET_ROOT.exists():
        print(f"Dataset root not found: {DATASET_ROOT}")
        print("Skipping pose extraction smoke test.")
        return

    keypoint_files = sorted(FEATURES_DIR.glob("*_keypoints.npy"))
    if not keypoint_files:
        print(f"No keypoint files found under: {FEATURES_DIR}")
        print("Skipping pose extraction smoke test.")
        return

    sample_path = keypoint_files[0]
    pose = np.load(sample_path)

    print(f"Testing on: {sample_path.name}")
    print(f"Extracted shape: {pose.shape}")

    if pose.ndim != 3:
        raise SystemExit(f"Unexpected keypoint tensor rank: {pose.ndim}")

    if pose.size == 0:
        raise SystemExit("Loaded keypoint tensor is empty")

    print(f"Sample keypoint[0,0,:]: {pose[0, 0, :]}")
    print("✓ Keypoint dataset smoke test passed")


if __name__ == "__main__":
    main()
