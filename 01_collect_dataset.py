"""
UGANDA SIGN LANGUAGE — Dataset Collector  (camera-fixed version)
MediaPipe 0.10+ compatible | Wayland fix built-in

Run:
  python3 01_collect_dataset.py --mode alphabet
  python3 01_collect_dataset.py --mode alphabet --camera 1   # if camera 0 fails
"""
import os, sys

# ── Wayland fix (must happen BEFORE any cv2 import) ─────────────────────
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import argparse
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hand_engine import HandEngine

SAMPLES_PER_CLASS = 100
DATA_DIR          = Path("data")
ALPHABET_CLASSES  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUMBER_CLASSES    = [str(i) for i in range(1, 11)]

SIGN_HINTS = {
    "A":"Closed fist, thumb on side","B":"Flat hand up, thumb tucked",
    "C":"Curved C shape","D":"Index up, others circle",
    "E":"Fingers curled, thumb under","F":"Index+thumb circle, rest up",
    "G":"Index+thumb point sideways","H":"Index+middle horizontal",
    "I":"Pinky only","J":"Pinky + draw J","K":"V up, thumb between",
    "L":"L shape","M":"Three fingers over thumb","N":"Two fingers over thumb",
    "O":"All fingers form O","P":"K shape pointing down",
    "Q":"G shape pointing down","R":"Index+middle crossed",
    "S":"Fist, thumb over fingers","T":"Thumb between index+middle",
    "U":"Index+middle together up","V":"Peace sign",
    "W":"Three fingers spread","X":"Index hooked",
    "Y":"Thumb+pinky out","Z":"Index draws Z",
    "1":"Index up","2":"V sign","3":"Thumb+index+middle",
    "4":"Four fingers","5":"Open hand","6":"Pinky+thumb touch",
    "7":"Ring+thumb touch","8":"Middle+thumb touch",
    "9":"Index+thumb circle","10":"Thumb up + shake",
}

BG=(18,18,28); GRN=(0,220,120); YLW=(0,210,255); RED=(60,60,255)
WHT=(240,240,240); GRY=(140,140,150); DARK=(30,30,50)


def open_camera(index=0):
    """Try to open camera, auto-retry with V4L2 backend if first attempt fails."""
    # Try default backend first
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if cap.isOpened():
        # Quick frame test
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  ✅ Camera {index} opened (default backend)")
            return cap

    cap.release()

    # Try V4L2 explicitly
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  ✅ Camera {index} opened (V4L2 backend)")
            return cap

    cap.release()
    return None


def collect_class(cap, engine, class_name, output_dir,
                  n_samples=SAMPLES_PER_CLASS):
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, raw_imgs = [], []
    start_t, countdown = time.time(), 3
    print(f"\n  ► '{class_name}'  —  {SIGN_HINTS.get(class_name,'')}")
    print(f"    SPACE=capture  N=skip  Q=quit")

    while len(samples) < n_samples:
        ret, frame = cap.read()
        if not ret:
            # Camera read failed — skip frame silently
            time.sleep(0.05)
            continue

        frame   = cv2.flip(frame, 1)
        lm, ann = engine.process(frame)

        canvas    = np.zeros((520, 700, 3), dtype=np.uint8)
        canvas[:] = BG

        canvas[80:410, 10:450] = cv2.resize(ann, (440, 330))
        cv2.rectangle(canvas, (10,80), (450,410),
                      GRN if lm is not None else RED, 2)

        elapsed = time.time() - start_t
        if elapsed < countdown:
            rem = int(countdown - elapsed) + 1
            cv2.putText(canvas, str(rem), (195,270),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, YLW, 8, cv2.LINE_AA)
        else:
            cv2.putText(canvas, f"Sign: {class_name}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, GRN, 3, cv2.LINE_AA)

        # Info panel
        px = 460
        cv2.rectangle(canvas, (px,80), (690,410), DARK, -1)
        hint  = SIGN_HINTS.get(class_name, "")
        words = hint.split()
        line, lines = "", []
        for w in words:
            if len(line)+len(w) < 22: line += (" " if line else "")+w
            else: lines.append(line); line = w
        if line: lines.append(line)
        for i, l in enumerate(lines[:4]):
            cv2.putText(canvas, l, (px+8, 110+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHT, 1)

        cv2.putText(canvas, f"{len(samples)}/{n_samples}",
                    (px+8, 258), cv2.FONT_HERSHEY_SIMPLEX, 1.1, GRN, 2)
        pct = len(samples) / n_samples
        cv2.rectangle(canvas, (px+8,280), (px+218,294), (40,42,60), -1)
        cv2.rectangle(canvas, (px+8,280), (px+8+int(210*pct),294), GRN, -1)

        sc = GRN if lm is not None else RED
        cv2.putText(canvas, "HAND OK" if lm is not None else "NO HAND",
                    (px+8, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 2)
        cv2.putText(canvas, "SPACE=capture", (px+8,360),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, GRN, 1)
        cv2.putText(canvas, "N=next  Q=quit", (px+8,382),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, GRY, 1)
        cv2.putText(canvas, f"Mode:{engine.mode}", (10,500),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRY, 1)

        cv2.imshow("USL Dataset Collector", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): return samples, True
        if key == ord('n'):
            print(f"    ⏭ Skipped (saved {len(samples)})")
            break
        if key == ord(' ') and elapsed >= countdown:
            if lm is not None:
                samples.append(lm.copy())
                raw_imgs.append(cv2.resize(frame, (64, 64)))
                print(f"    ✓ {len(samples)}/{n_samples}", end='\r')
            else:
                print("    ✗ No hand detected", end='\r')

    if samples:
        np.save(output_dir / f"{class_name}_landmarks.npy",
                np.array(samples, dtype=np.float32))
        np.save(output_dir / f"{class_name}_images.npy",
                np.array(raw_imgs, dtype=np.uint8))
        print(f"\n  ✅ Saved {len(samples)} samples → {output_dir}")

    return samples, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["alphabet","numbers","both"], default="both")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_CLASS)
    parser.add_argument("--camera",  type=int, default=0,
                        help="Camera index (try 0, 1, 2 if you get timeouts)")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if args.mode == "alphabet":
        classes, out_root = ALPHABET_CLASSES, DATA_DIR/"alphabets"
    elif args.mode == "numbers":
        classes, out_root = NUMBER_CLASSES, DATA_DIR/"numbers"
    else:
        classes  = ALPHABET_CLASSES + NUMBER_CLASSES
        out_root = DATA_DIR

    print("=" * 58)
    print("  UGANDA SIGN LANGUAGE — Dataset Collector")
    print("=" * 58)
    print(f"  Mode    : {args.mode}  |  Classes: {len(classes)}")
    print(f"  Samples : {args.samples} per class")
    print(f"  Camera  : index {args.camera}")
    print("=" * 58)

    engine = HandEngine()
    cap    = open_camera(args.camera)

    if cap is None:
        print(f"\n  ❌ Cannot open camera {args.camera}")
        print("  → Run: python3 diagnose_camera.py  to find your working camera")
        print("  → Try: python3 01_collect_dataset.py --camera 1")
        engine.close()
        return

    try:
        for cls in classes:
            if args.mode == "both":
                sub    = "alphabets" if cls in ALPHABET_CLASSES else "numbers"
                outdir = DATA_DIR / sub / cls
            else:
                outdir = out_root / cls

            _, quit_sig = collect_class(
                cap, engine, cls, outdir, n_samples=args.samples
            )
            if quit_sig:
                print("\n⚠ Stopped early.")
                break
    finally:
        cap.release()
        engine.close()
        cv2.destroyAllWindows()

    # Print summary
    for subset in ["alphabets", "numbers"]:
        d = DATA_DIR / subset
        if d.exists():
            idx = {}
            for cd in sorted(d.iterdir()):
                if cd.is_dir():
                    for f in cd.glob("*_landmarks.npy"):
                        arr = np.load(f)
                        lbl = f.stem.replace("_landmarks", "")
                        idx[lbl] = {"n_samples": len(arr)}
            if idx:
                total = sum(v["n_samples"] for v in idx.values())
                print(f"  {subset}: {len(idx)} classes, {total} samples")

    print("\n✅ Done! Next: python3 02_eda_analysis.py --mode", args.mode)


if __name__ == "__main__":
    main()
