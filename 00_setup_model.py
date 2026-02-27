"""
=============================================================
USL PROJECT — MODEL SETUP  (Run this ONCE with internet)
=============================================================
Downloads the MediaPipe hand_landmarker.task file required
by MediaPipe 0.10+ (Tasks API).

  python3 00_setup_model.py

The file is ~25MB and saved to: models/hand_landmarker.task
After this, all other scripts work fully offline.

If you have NO internet, the scripts will automatically fall
back to skin-color + contour hand detection (less accurate
but fully functional without downloading anything).
=============================================================
"""

import urllib.request
import pathlib
import sys
import hashlib

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = pathlib.Path("models/hand_landmarker.task")


def download_with_progress(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            mb_done  = downloaded / 1024 / 1024
            mb_total = total_size  / 1024 / 1024
            print(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB",
                  end="", flush=True)

    print(f"  Downloading hand_landmarker.task ...")
    print(f"  URL: {url}\n")
    urllib.request.urlretrieve(url, dest, reporthook)
    print(f"\n\n  ✅ Saved to: {dest}  ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    print("=" * 60)
    print("  USL Project — MediaPipe Model Setup")
    print("=" * 60)

    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
        print(f"  ✅ Model already present: {MODEL_PATH}  ({size_mb:.1f} MB)")
        print("  No action needed.\n")
        return

    print(f"  Target: {MODEL_PATH.resolve()}\n")
    try:
        download_with_progress(MODEL_URL, MODEL_PATH)
        print("\n  ✅ Setup complete! You can now run all USL scripts offline.")
        print("  Next: python3 01_collect_dataset.py --mode alphabet")
    except Exception as e:
        print(f"\n  ❌ Download failed: {e}")
        print("\n  If you have no internet access, the USL scripts will")
        print("  automatically use SKIN-COLOR DETECTION mode as fallback.")
        print("  This works without any model file — just less precise.")
        print("\n  To enable full MediaPipe accuracy later:")
        print(f"    wget '{MODEL_URL}' -O {MODEL_PATH}")
        sys.exit(1)


if __name__ == "__main__":
    main()
