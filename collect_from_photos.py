"""
=============================================================
USL — COLLECT DATASET FROM PHOTOS  (no webcam needed)
=============================================================
Take hand-sign photos on your phone → transfer to PC →
this script extracts MediaPipe hand landmarks from them.

STEP 1 — Create the folder structure:
  python3 collect_from_photos.py --mode alphabet --create-folders

STEP 2 — Add photos to the folders:
  data/photos/alphabets/A/photo1.jpg
  data/photos/alphabets/B/photo1.jpg  ... etc.
  (Any .jpg / .png files work, file names don't matter)

STEP 3 — Extract landmarks:
  python3 collect_from_photos.py --mode alphabet
  python3 collect_from_photos.py --mode numbers
  python3 collect_from_photos.py --mode both

TIPS FOR GOOD PHOTOS:
  - Bright, even lighting (natural light is best)
  - Plain background (white wall, desk, etc.)
  - Hand fills most of the frame
  - Take 20-50 photos per sign from different angles
  - Avoid blurry / dark images

AFTER RUNNING:
  python3 02_eda_analysis.py --mode alphabet
  python3 03_train_models.py --mode alphabet
  python3 04_app.py --mode alphabet  (needs webcam for live scoring)
=============================================================
"""
import os, sys, cv2, numpy as np, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hand_engine import HandEngine

DATA_DIR         = Path("data")
ALPHABET_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUMBER_CLASSES   = [str(i) for i in range(1, 11)]
IMG_EXTS         = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def process_photo_folder(engine, class_name, photo_dir, output_dir, show=False):
    """Extract landmarks from all photos in a folder."""
    output_dir.mkdir(parents=True, exist_ok=True)

    photos = []
    for ext in IMG_EXTS:
        photos.extend(photo_dir.glob(ext))
    photos = sorted(set(photos))

    if not photos:
        return 0

    print(f"\n  [{class_name}]  {len(photos)} photo(s) found")
    samples, raw_imgs, skipped = [], [], 0

    for photo_path in photos:
        img = cv2.imread(str(photo_path))
        if img is None:
            print(f"    ✗ Cannot read: {photo_path.name}")
            skipped += 1
            continue

        # Resize large images for speed
        h, w = img.shape[:2]
        if max(h, w) > 1080:
            scale = 1080 / max(h, w)
            img   = cv2.resize(img, (int(w*scale), int(h*scale)))

        lm, ann = engine.process(img)

        if lm is not None:
            samples.append(lm.copy())
            raw_imgs.append(cv2.resize(img, (64, 64)))
            status = "✓"
        else:
            skipped += 1
            status = "✗ no hand"

        print(f"    {status}  {photo_path.name}")

        # Optional: show annotated image for 0.5s
        if show and lm is not None:
            cv2.imshow(f"Processing: {class_name}", cv2.resize(ann, (480, 360)))
            cv2.waitKey(500)

    if show:
        cv2.destroyAllWindows()

    if samples:
        np.save(output_dir / f"{class_name}_landmarks.npy",
                np.array(samples, dtype=np.float32))
        np.save(output_dir / f"{class_name}_images.npy",
                np.array(raw_imgs, dtype=np.uint8))
        print(f"  ✅ Saved {len(samples)} samples  ({skipped} skipped)")
    else:
        print(f"  ❌ No valid hand samples extracted for '{class_name}'")
        print(f"     Tips: better lighting, hand fills frame, clear background")

    return len(samples)


def create_folder_structure(mode):
    """Create empty photo folders with README."""
    if mode in ("alphabet", "both"):
        base = DATA_DIR / "photos" / "alphabets"
        base.mkdir(parents=True, exist_ok=True)
        for cls in ALPHABET_CLASSES:
            (base / cls).mkdir(exist_ok=True)
        _write_readme(base, "alphabets", "alphabet")
        print(f"  ✅ Alphabet folders: {base.resolve()}")

    if mode in ("numbers", "both"):
        base = DATA_DIR / "photos" / "numbers"
        base.mkdir(parents=True, exist_ok=True)
        for cls in NUMBER_CLASSES:
            (base / cls).mkdir(exist_ok=True)
        _write_readme(base, "numbers", "numbers")
        print(f"  ✅ Numbers folders:  {base.resolve()}")


def _write_readme(base, subset, mode):
    (base / "README.txt").write_text(f"""\
USL PHOTO DATASET — {subset.upper()}
===========================================
Add hand-sign photos to each subfolder.

Folder structure:
  {base}/{subset[0].upper()}/photo1.jpg
  {base}/{subset[0].upper()}/photo2.jpg
  ...

Tips for good photos:
  ✓ Bright lighting (daylight near a window)
  ✓ Plain background (white/grey wall)
  ✓ Hand fills most of the frame
  ✓ 20-50 photos per sign
  ✓ Mix of angles and distances
  ✗ Avoid blurry, dark, or cluttered backgrounds

Supported formats: .jpg  .jpeg  .png

After adding photos:
  python3 collect_from_photos.py --mode {mode}
  python3 02_eda_analysis.py --mode {mode}
  python3 03_train_models.py --mode {mode}
""")


def print_photo_counts(base, classes):
    """Show how many photos exist per class."""
    has_photos = False
    for cls in classes:
        d = base / cls
        if d.exists():
            n = sum(len(list(d.glob(ext))) for ext in IMG_EXTS)
            if n > 0:
                bar = "█" * min(n, 30) + f"  {n}"
                print(f"    {cls:3}: {bar}")
                has_photos = True
    if not has_photos:
        print("    (no photos yet)")
    return has_photos


def main():
    parser = argparse.ArgumentParser(
        description="Extract USL hand landmarks from photo files"
    )
    parser.add_argument("--mode", choices=["alphabet","numbers","both"],
                        default="alphabet")
    parser.add_argument("--create-folders", action="store_true",
                        help="Create empty folder structure then exit")
    parser.add_argument("--show", action="store_true",
                        help="Show annotated image during processing")
    args = parser.parse_args()

    print("=" * 58)
    print("  USL — Photo Dataset Collector")
    print("=" * 58)

    if args.create_folders:
        print("\n  Creating folder structure...")
        create_folder_structure(args.mode)
        print("""
  ✅ Done! Now:
     1. Transfer hand-sign photos to the folders above
     2. Run: python3 collect_from_photos.py --mode """ + args.mode)
        return

    # Show current photo counts
    modes = ["alphabet","numbers"] if args.mode=="both" else [args.mode]
    for mode in modes:
        subset = "alphabets" if mode=="alphabet" else "numbers"
        base   = DATA_DIR / "photos" / subset
        classes = ALPHABET_CLASSES if mode=="alphabet" else NUMBER_CLASSES

        print(f"\n  Photo counts — {mode}:")
        if base.exists():
            has = print_photo_counts(base, classes)
            if not has:
                print(f"\n  No photos found in: {base.resolve()}")
                print(f"  Create folders first:")
                print(f"    python3 collect_from_photos.py --mode {mode} --create-folders")
                continue
        else:
            print(f"  Folder not found: {base}")
            print(f"  Create it: python3 collect_from_photos.py --mode {mode} --create-folders")
            continue

    # Process
    engine       = HandEngine()
    total        = 0
    DATA_DIR.mkdir(exist_ok=True)

    for mode in modes:
        subset  = "alphabets" if mode=="alphabet" else "numbers"
        classes = ALPHABET_CLASSES if mode=="alphabet" else NUMBER_CLASSES
        base    = DATA_DIR / "photos" / subset

        if not base.exists():
            continue

        print(f"\n{'─'*50}")
        print(f"  Extracting landmarks from {mode} photos...")
        print(f"{'─'*50}")

        for cls in classes:
            photo_dir  = base / cls
            output_dir = DATA_DIR / subset / cls
            if not photo_dir.exists():
                continue
            has_photos = any(
                any(photo_dir.glob(ext)) for ext in IMG_EXTS
            )
            if has_photos:
                n = process_photo_folder(
                    engine, cls, photo_dir, output_dir, show=args.show
                )
                total += n

    engine.close()

    print(f"\n{'='*58}")
    print(f"  Total samples extracted: {total}")

    if total > 0:
        print(f"\n  ✅ Dataset ready!")
        print(f"  Next steps:")
        print(f"    python3 02_eda_analysis.py --mode {args.mode}")
        print(f"    python3 03_train_models.py --mode {args.mode}")
        print(f"    python3 04_app.py --mode {args.mode}  (needs webcam)")
    else:
        print(f"\n  ⚠  No samples collected.")
        print(f"  → Make sure photos are in: data/photos/alphabets/A/")
        print(f"  → Run: python3 collect_from_photos.py --mode {args.mode} --create-folders")


if __name__ == "__main__":
    main()
