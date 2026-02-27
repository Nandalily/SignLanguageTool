"""
UGANDA SIGN LANGUAGE — Live Training App  (camera-fixed version)
MediaPipe 0.10+ compatible | Wayland fix built-in

Run:
  python3 04_app.py --mode alphabet
  python3 04_app.py --mode alphabet --camera 1   # if camera 0 fails
"""
import os, sys

# ── Wayland fix (must happen BEFORE any cv2 import) ──────────────────────
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import pickle
import webbrowser
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hand_engine import HandEngine

MODEL_DIR        = Path("models")
ALPHABET_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUMBER_CLASSES   = [str(i) for i in range(1, 11)]

YOUTUBE_LINKS = {
    **{l: f"https://www.youtube.com/results?search_query=Uganda+sign+language+letter+{l}"
       for l in ALPHABET_CLASSES},
    **{str(n): f"https://www.youtube.com/results?search_query=Uganda+sign+language+number+{n}"
       for n in range(1, 11)},
}

SIGN_HINTS = {
    "A":"Closed fist, thumb on side",
    "B":"Flat hand up, fingers together, thumb tucked",
    "C":"Curved hand — C shape",
    "D":"Index up, thumb + fingers form a circle",
    "E":"Fingers curled, thumb tucked under",
    "F":"Index + thumb circle, other fingers up",
    "G":"Index + thumb point horizontally",
    "H":"Index + middle extended horizontally",
    "I":"Pinky only extended",
    "J":"Pinky extended, draw a J",
    "K":"V up, thumb between index + middle",
    "L":"L-shape: index up, thumb out",
    "M":"Three fingers folded over thumb",
    "N":"Two fingers folded over thumb",
    "O":"All fingers curve to touch thumb — O shape",
    "P":"K-shape pointing downward",
    "Q":"G-shape pointing downward",
    "R":"Index + middle fingers crossed",
    "S":"Closed fist, thumb over fingers",
    "T":"Thumb between index + middle finger",
    "U":"Index + middle together pointing up",
    "V":"Index + middle spread — V / peace sign",
    "W":"Index + middle + ring fingers spread",
    "X":"Index finger hooked",
    "Y":"Thumb + pinky out — hang loose",
    "Z":"Index draws a Z in the air",
    "1":"Index finger pointing up",
    "2":"Index + middle — V sign",
    "3":"Thumb + index + middle up",
    "4":"Four fingers up, thumb tucked",
    "5":"All five fingers spread open",
    "6":"Pinky + thumb touch, others up",
    "7":"Ring + thumb touch, others up",
    "8":"Middle + thumb touch, others up",
    "9":"Index + thumb circle, others up",
    "10":"Thumb up + shake",
}

# Colours (BGR)
BG   = (18, 18, 28)
GRN  = (0, 220, 120)
YLW  = (0, 210, 255)
RED  = (60, 60, 230)
WHT  = (240, 240, 240)
GRY  = (140, 140, 155)
DARK = (28, 28, 48)
ACNT = (100, 230, 180)

W, H, PNL_W = 1280, 720, 400
CAM_W = W - PNL_W
CAM_H = H


def open_camera(index=0):
    """Reliably open camera with fallback backends."""
    for backend, name in [
        (cv2.CAP_ANY,    "auto"),
        (cv2.CAP_V4L2,   "V4L2"),
        (cv2.CAP_FFMPEG, "FFmpeg"),
    ]:
        try:
            cap = cv2.VideoCapture(index, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"  ✅ Camera {index} opened [{name} backend]")
                    return cap
            cap.release()
        except Exception:
            pass
    return None


def load_grader(mode):
    sub = "alphabet" if mode == "alphabet" else "numbers"
    for p in [
        MODEL_DIR / sub / "supervised_grader.pkl",
        Path("reports/training") / sub / "supervised_grader.pkl",
    ]:
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
    print(f"  ⚠ No grader model for '{mode}' — run 03_train_models.py first")
    return None


def grade(model, lm, target):
    if model is None or lm is None:
        return 10
    try:
        le  = model["label_encoder"]
        Xs  = model["scaler"].transform(lm.reshape(1, -1))
        Xr  = model["pca"].transform(Xs)
        p   = model["clf"].predict_proba(Xr)[0]
        ti  = le.transform([target])[0]
        s   = max(10, min(100, int(p[ti] * 100 // 10) * 10))
        return s if s > 0 else 10
    except Exception:
        return 10


def t(img, text, x, y, sc=0.6, color=WHT, thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                sc, color, thick, cv2.LINE_AA)


def sc_col(s):
    return GRN if s >= 70 else (YLW if s >= 40 else RED)


def sc_lbl(s):
    if s >= 90: return "EXCELLENT!"
    if s >= 70: return "GOOD JOB!"
    if s >= 50: return "KEEP TRYING"
    return "NEEDS WORK"


def wrap(text, n=32):
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 <= n:
            line += (" " if line else "") + w
        else:
            lines.append(line); line = w
    if line: lines.append(line)
    return lines


def draw_grid(img, classes, sel, x0, y0):
    cols = 9 if len(classes) > 10 else 5
    cw, ch, pad = 40, 36, 5
    for i, cls in enumerate(classes):
        cx = x0 + (i % cols) * (cw + pad)
        cy = y0 + (i // cols) * (ch + pad)
        bg = ACNT if cls == sel else (45, 48, 68)
        tc = BG   if cls == sel else WHT
        cv2.rectangle(img, (cx, cy), (cx+cw, cy+ch), bg, -1)
        cv2.rectangle(img, (cx, cy), (cx+cw, cy+ch), (70, 72, 95), 1)
        cv2.putText(img, cls,
                    (cx + (9 if len(cls) == 1 else 4), cy + ch - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, tc, 2, cv2.LINE_AA)


def score_bar(img, s, x, y, bw=260, bh=20):
    cv2.rectangle(img, (x, y), (x+bw, y+bh), (40, 42, 60), -1)
    cv2.rectangle(img, (x, y), (x+int(bw*s/100), y+bh), sc_col(s), -1)
    t(img, f"{s}%", x+bw+10, y+bh-3, 0.7, sc_col(s), 2)


class App:
    def __init__(self, mode, camera=0):
        self.mode    = mode
        self.camera  = camera
        self.classes = ALPHABET_CLASSES if mode == "alphabet" else NUMBER_CLASSES
        self.sel     = self.classes[0]
        self.state   = "SELECT"
        self.engine  = HandEngine()
        self.grader  = load_grader(mode)
        self.score   = None
        self.lm      = None
        self.hand_ok = False
        self.captured = None
        self.fb      = []

    def _fb(self, score):
        hint = SIGN_HINTS.get(self.sel, "")
        if score >= 80:
            return ["Great form! Matches the correct sign well."]
        if score >= 50:
            return ["Good effort! Remember:"] + wrap(hint, 32)[:2]
        return ["Try again. Correct form:"] + wrap(hint, 32)[:2]

    def run(self):
        cap = open_camera(self.camera)
        if cap is None:
            print(f"\n  ❌ Cannot open camera {self.camera}")
            print("  → Run: python3 diagnose_camera.py")
            print(f"  → Try: python3 04_app.py --camera 1")
            self.engine.close()
            return

        cv2.namedWindow("USL Trainer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("USL Trainer", W, H)

        last_raw = None
        print(f"\n🤟 USL Trainer — {self.mode.upper()}")
        print("   SPACE=capture  R=reset  N=next  O=YouTube  Q=quit\n")

        while True:
            canvas    = np.zeros((H, W, 3), dtype=np.uint8)
            canvas[:] = BG

            ret, frame = cap.read()
            ann = None

            if ret and frame is not None and self.state == "CAMERA":
                frame    = cv2.flip(frame, 1)
                self.lm, ann = self.engine.process(frame)
                self.hand_ok = self.lm is not None
                last_raw     = frame.copy()

            # ── SELECT ──────────────────────────────────────────────────
            if self.state == "SELECT":
                cv2.rectangle(canvas, (0,0), (W,62), DARK, -1)
                t(canvas, "UGANDA SIGN LANGUAGE TRAINER", 20, 40, 0.9, ACNT, 2)
                t(canvas, f"Mode: {self.mode.upper()}", W-210, 40, 0.6, GRY, 1)
                t(canvas, "Choose a sign — press key or click, then ENTER",
                  20, 110, 0.52, GRY, 1)
                draw_grid(canvas, self.classes, self.sel, 22, 130)

                bx, by = 22, 360
                cv2.rectangle(canvas, (bx,by), (bx+600,by+210), DARK, -1)
                cv2.rectangle(canvas, (bx,by), (bx+600,by+210), (55,58,80), 1)
                cv2.putText(canvas, self.sel, (bx+18, by+72),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.8, ACNT, 6, cv2.LINE_AA)
                for i, l in enumerate(wrap(SIGN_HINTS.get(self.sel,""), 40)[:3]):
                    t(canvas, l, bx+120, by+36+i*24, 0.58, WHT, 1)
                t(canvas, "Press ENTER to open camera",
                  bx+18, by+178, 0.65, GRN, 2)

            # ── CAMERA ──────────────────────────────────────────────────
            elif self.state == "CAMERA":
                if ann is not None:
                    canvas[:CAM_H, :CAM_W] = cv2.resize(ann, (CAM_W, CAM_H))
                elif not ret:
                    t(canvas, "Camera not responding...", 20, CAM_H//2,
                      1.0, YLW, 2)

                p = 18
                for xs, ys in [(p,p),(CAM_W-p-50,p),(p,CAM_H-p),(CAM_W-p-50,CAM_H-p)]:
                    cv2.line(canvas, (xs,ys), (xs+50,ys), GRN, 2)
                    dy = 50 if ys == p else -50
                    cv2.line(canvas, (xs,ys), (xs,ys+dy), GRN, 2)

                hc = GRN if self.hand_ok else RED
                cv2.circle(canvas, (CAM_W-35, 35), 11, hc, -1)
                t(canvas, "HAND" if self.hand_ok else "NO HAND",
                  CAM_W-90, 42, 0.45, hc, 1)

                px = CAM_W
                cv2.rectangle(canvas, (px,0), (W,H), DARK, -1)
                cv2.line(canvas, (px,0), (px,H), ACNT, 1)
                t(canvas, "SIGN TO PRACTICE", px+14, 38, 0.52, GRY, 1)
                cv2.putText(canvas, self.sel, (px+50, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.2, ACNT, 8, cv2.LINE_AA)
                for i, l in enumerate(wrap(SIGN_HINTS.get(self.sel,""), 28)[:3]):
                    t(canvas, l, px+14, 178+i*22, 0.5, GRY, 1)
                cv2.line(canvas, (px+14,240), (W-14,240), (45,48,68), 1)
                t(canvas, "Position hand in frame",  px+14, 270, 0.52, WHT, 1)
                t(canvas, "SPACE / ENTER = capture", px+14, 370, 0.55, GRN, 2)
                t(canvas, "R = menu    Q = quit",    px+14, 400, 0.5,  GRY, 1)
                t(canvas, f"Engine: {self.engine.mode}", px+14, H-16, 0.42, GRY, 1)

            # ── RESULT ──────────────────────────────────────────────────
            elif self.state == "RESULT":
                sc = sc_col(self.score)
                cv2.rectangle(canvas, (0,0), (W,65), DARK, -1)
                t(canvas, f"RESULT  —  Sign '{self.sel}'", 20, 42, 0.95, ACNT, 2)

                if self.captured is not None:
                    ih = int(CAM_H * 0.6)
                    iw = int(ih * CAM_W / CAM_H)
                    canvas[80:80+ih, 16:16+iw] = cv2.resize(self.captured, (iw,ih))
                    cv2.rectangle(canvas, (16,80), (16+iw,80+ih), sc, 3)

                px = CAM_W
                cv2.rectangle(canvas, (px,0), (W,H), DARK, -1)
                cv2.line(canvas, (px,0), (px,H), ACNT, 1)
                t(canvas, "YOUR SCORE", px+14, 100, 0.55, GRY, 1)
                cv2.putText(canvas, f"{self.score}%", (px+20,210),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.8, sc, 8, cv2.LINE_AA)
                t(canvas, sc_lbl(self.score), px+20, 248, 0.82, sc, 2)
                score_bar(canvas, self.score, px+14, 265)
                cv2.line(canvas, (px+14,310), (W-14,310), (45,48,68), 1)
                t(canvas, "FEEDBACK", px+14, 340, 0.52, GRY, 1)
                for i, l in enumerate(self.fb[:4]):
                    t(canvas, l, px+14, 365+i*24, 0.5, WHT, 1)

                yy = 475
                cv2.rectangle(canvas, (px+14,yy), (W-14,yy+38), (28,28,75), -1)
                cv2.rectangle(canvas, (px+14,yy), (W-14,yy+38), (70,70,180), 1)
                t(canvas, "O = Open YouTube Tutorial",
                  px+22, yy+24, 0.52, (120,130,255), 1)
                cv2.rectangle(canvas, (px+14,528), (px+164,565), (25,80,45), -1)
                t(canvas, "R = Try Again", px+22, 551, 0.52, GRN, 1)
                cv2.rectangle(canvas, (px+174,528), (W-14,565), (30,30,75), -1)
                t(canvas, "N = Next Sign", px+182, 551, 0.52, ACNT, 1)
                t(canvas, "Q / ESC = Quit", px+14, 608, 0.48, (120,80,80), 1)

            cv2.imshow("USL Trainer", canvas)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif self.state == "SELECT":
                if key in (13, 32):
                    self.state = "CAMERA"
                    self.lm    = None
                elif 32 <= key < 128:
                    ch = chr(key).upper()
                    if ch in self.classes:
                        self.sel = ch

            elif self.state == "CAMERA":
                if key in (13, 32):
                    if self.lm is not None:
                        self.score    = grade(self.grader, self.lm, self.sel)
                        self.fb       = self._fb(self.score)
                        self.captured = (last_raw.copy()
                                         if last_raw is not None else None)
                        self.state    = "RESULT"
                        print(f"   '{self.sel}' → {self.score}%  {sc_lbl(self.score)}")
                    else:
                        print("   ✗ No hand detected — try again")
                elif key == ord('r'):
                    self.state = "SELECT"

            elif self.state == "RESULT":
                if key == ord('r'):
                    self.state    = "CAMERA"
                    self.score    = None
                    self.captured = None
                elif key == ord('n'):
                    idx = self.classes.index(self.sel)
                    self.sel   = self.classes[(idx+1) % len(self.classes)]
                    self.state = "SELECT"
                    self.score = None
                elif key == ord('o'):
                    url = YOUTUBE_LINKS.get(self.sel, "")
                    if url:
                        webbrowser.open(url)
                        print(f"   🎥 {url}")

        cap.release()
        self.engine.close()
        cv2.destroyAllWindows()
        print("\n👋 USL Trainer closed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["alphabet","numbers","both"],
                        default="alphabet")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (try 0, 1, 2 if you get timeouts)")
    args = parser.parse_args()

    mode = args.mode
    if mode == "both":
        mode = "numbers" if input("1=Alphabet  2=Numbers: ").strip()=="2" \
               else "alphabet"

    App(mode=mode, camera=args.camera).run()


if __name__ == "__main__":
    main()
