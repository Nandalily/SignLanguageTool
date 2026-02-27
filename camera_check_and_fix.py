"""
=============================================================
USL — CAMERA CHECK + PHOTO-BASED DATA COLLECTION FALLBACK
=============================================================
Run this when webcam is unavailable.

  python3 camera_check_and_fix.py

This script:
  1. Shows exactly why your camera isn't working
  2. Lets you collect data using PHOTO FILES instead of live webcam
  3. Tells you how to fix the camera properly

Photo collection: take hand-sign photos on your phone,
transfer them, and this script will extract landmarks from them.
=============================================================
"""
import os, sys, pathlib, subprocess

HERE = pathlib.Path(__file__).parent
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

print("=" * 60)
print("  USL — Camera Diagnostic")
print("=" * 60)

# ── Check /dev/video* ────────────────────────────────────────
videos = sorted(pathlib.Path("/dev").glob("video*"))
print(f"\n[1] Video devices: {len(videos)} found")
if not videos:
    print("    ❌ /dev/video0 does not exist")
    print("    This means either:")
    print("      a) No webcam is connected")
    print("      b) Webcam not recognised by kernel")
    print("      c) Running in a VM with no USB passthrough")
else:
    for v in videos:
        print(f"    ✅ {v}")

# ── lsusb camera check ───────────────────────────────────────
print("\n[2] USB camera devices:")
try:
    result = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=5)
    cam_kw = ["camera","webcam","video","uvc","logitech","microsoft",
              "trust","creative","chicony","realtek","sonix","bison","sunplus"]
    found = [l for l in result.stdout.splitlines()
             if any(k in l.lower() for k in cam_kw)]
    if found:
        for l in found: print(f"    📷 {l}")
    else:
        print("    ❌ No camera USB device detected")
        print("    → Connect a USB webcam and re-run")
except Exception:
    print("    (lsusb not available)")

# ── VM check ─────────────────────────────────────────────────
print("\n[3] Environment:")
print(f"    Session: {os.environ.get('XDG_SESSION_TYPE','?')}")
print(f"    Display: {os.environ.get('DISPLAY','not set')}")
try:
    r = subprocess.run(["systemd-detect-virt"], capture_output=True, text=True, timeout=3)
    virt = r.stdout.strip()
    if virt and virt != "none":
        print(f"    ⚠  Running in VM: {virt}")
        print("    → Enable USB passthrough in VM settings to use webcam")
    else:
        print("    ✅ Not running in a VM")
except Exception:
    pass

# ── Solutions ────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SOLUTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION A — Connect a USB webcam (recommended):
  1. Plug in any USB webcam
  2. Run: ls /dev/video*   (should show /dev/video0)
  3. Run: python3 01_collect_dataset.py --mode alphabet

OPTION B — Use phone as webcam (DroidCam):
  1. Install DroidCam on Android phone (free)
  2. Install client: sudo apt install droidcam
  3. Connect phone via USB
  4. Run droidcam on both phone and PC
  5. Run: python3 01_collect_dataset.py --mode alphabet

OPTION C — Collect data from PHOTOS (no webcam needed):
  1. Take photos of your hand signs on your phone
  2. Transfer photos to: data/photos/A/photo1.jpg, etc.
  3. Run: python3 collect_from_photos.py --mode alphabet
  (This script will be created below)

OPTION D — Fix VM USB passthrough (if in VirtualBox):
  Settings → USB → Add USB Device Filter → select your webcam
  Then restart the VM.
""")

# ── Create collect_from_photos.py ────────────────────────────
PHOTO_SCRIPT = '''"""
USL — Collect dataset from PHOTO FILES (no webcam needed)
Usage:
  1. Create folders:  data/photos/A/  data/photos/B/  etc.
  2. Put hand-sign photos in each folder (jpg/png, any size)
  3. Run: python3 collect_from_photos.py --mode alphabet
"""
import os, sys, cv2, numpy as np, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from hand_engine import HandEngine

DATA_DIR         = Path("data")
ALPHABET_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUMBER_CLASSES   = [str(i) for i in range(1, 11)]


def process_photo_folder(engine, class_name, photo_dir, output_dir):
    """Extract landmarks from all photos in a folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    photos = list(photo_dir.glob("*.jpg")) + list(photo_dir.glob("*.png")) \\
           + list(photo_dir.glob("*.jpeg")) + list(photo_dir.glob("*.JPG"))

    if not photos:
        print(f"  ⚠  No photos in {photo_dir}")
        return 0

    samples, raw_imgs = [], []
    for photo_path in sorted(photos):
        img = cv2.imread(str(photo_path))
        if img is None:
            print(f"    ✗ Cannot read {photo_path.name}")
            continue
        lm, ann = engine.process(img)
        if lm is not None:
            samples.append(lm.copy())
            raw_imgs.append(cv2.resize(img, (64, 64)))
            print(f"    ✓ {photo_path.name} — landmarks extracted")
        else:
            print(f"    ✗ {photo_path.name} — no hand detected (check lighting/angle)")

    if samples:
        np.save(output_dir / f"{class_name}_landmarks.npy",
                np.array(samples, dtype=np.float32))
        np.save(output_dir / f"{class_name}_images.npy",
                np.array(raw_imgs, dtype=np.uint8))
        print(f"  ✅ {class_name}: saved {len(samples)}/{len(photos)} samples")
    return len(samples)


def create_sample_structure(mode):
    """Create the folder structure with a README."""
    classes = ALPHABET_CLASSES if mode == "alphabet" else NUMBER_CLASSES
    subset  = "alphabets" if mode == "alphabet" else "numbers"
    base    = DATA_DIR / "photos" / subset
    base.mkdir(parents=True, exist_ok=True)

    for cls in classes:
        d = base / cls
        d.mkdir(exist_ok=True)

    readme = base / "README.txt"
    readme.write_text(f"""
HOW TO COLLECT PHOTOS FOR USL DATASET
======================================
For each letter/number folder, add CLEAR photos of your hand sign.

Tips for good photos:
  - Good lighting (natural light works best)
  - Hand clearly visible against contrasting background
  - Try 10-30 different angles / positions per sign
  - File names don't matter (photo1.jpg, img_001.jpg, etc.)

Folder structure:
  data/photos/{subset}/A/photo1.jpg
  data/photos/{subset}/A/photo2.jpg
  data/photos/{subset}/B/photo1.jpg
  ...

After adding photos, run:
  python3 collect_from_photos.py --mode {mode}
""")
    print(f"\\n  ✅ Created folder structure at: {base.resolve()}")
    print(f"  ✅ README written at: {readme}")
    print(f"\\n  Add photos to: {base.resolve()}/[A-Z]/")
    print(f"  Then run: python3 collect_from_photos.py --mode {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["alphabet","numbers","both"],
                        default="alphabet")
    parser.add_argument("--create-folders", action="store_true",
                        help="Just create the folder structure")
    args = parser.parse_args()

    if args.create_folders:
        for m in (["alphabet","numbers"] if args.mode=="both" else [args.mode]):
            create_sample_structure(m)
        return

    DATA_DIR.mkdir(exist_ok=True)
    modes = ["alphabet","numbers"] if args.mode=="both" else [args.mode]
    engine = HandEngine()
    total_samples = 0

    for mode in modes:
        classes = ALPHABET_CLASSES if mode=="alphabet" else NUMBER_CLASSES
        subset  = "alphabets"      if mode=="alphabet" else "numbers"
        photo_base = DATA_DIR / "photos" / subset

        if not photo_base.exists():
            print(f"\\n⚠  Photo folder not found: {photo_base}")
            print(f"   Create it with: python3 collect_from_photos.py --mode {mode} --create-folders")
            continue

        print(f"\\n{'─'*50}")
        print(f"  Processing {mode} photos from: {photo_base}")
        print(f"{'─'*50}")

        for cls in classes:
            photo_dir  = photo_base / cls
            output_dir = DATA_DIR / subset / cls
            if photo_dir.exists() and any(photo_dir.glob("*.jpg")) or \\
               photo_dir.exists() and any(photo_dir.glob("*.png")):
                print(f"\\n  [{cls}]")
                n = process_photo_folder(engine, cls, photo_dir, output_dir)
                total_samples += n
            else:
                pass  # silently skip empty folders

    engine.close()
    print(f"\\n{'='*50}")
    print(f"  Photo collection complete: {total_samples} samples extracted")
    print(f"  Next: python3 02_eda_analysis.py --mode {args.mode}")
    print(f"  Then: python3 03_train_models.py --mode {args.mode}")


if __name__ == "__main__":
    main()
'''

photo_script_path = HERE / "collect_from_photos.py"
photo_script_path.write_text(PHOTO_SCRIPT)
print(f"\n✅ Created: {photo_script_path}")
print("\n   Use this to collect data without a webcam:")
print("   python3 collect_from_photos.py --mode alphabet --create-folders")
print("   (Creates the folder structure, then add photos to each folder)")
