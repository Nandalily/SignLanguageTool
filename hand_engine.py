"""
=============================================================
USL PROJECT — HAND DETECTION ENGINE  (shared module)
=============================================================
Provides a unified hand landmark extractor that works with:

  MODE A — MediaPipe Tasks API (requires hand_landmarker.task)
           Full 21-landmark detection. Best accuracy.

  MODE B — Skin-color + Contour fallback (no model file needed)
           Estimates 21 pseudo-landmarks from hand contour.
           Works completely offline, less precise.

Import in other scripts:
  from hand_engine import HandEngine
=============================================================
"""

import cv2
import numpy as np
import pathlib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = pathlib.Path("models/hand_landmarker.task")

# Hand connections for drawing (21-landmark topology)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20), # pinky
    (5,9),(9,13),(13,17),           # palm
]


class HandEngine:
    """
    Unified hand landmark detector.
    Automatically picks Tasks API if model file is present,
    otherwise uses skin-color fallback.
    """

    def __init__(self):
        self.mode = None
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        if MODEL_PATH.exists():
            try:
                base_opts = mp_python.BaseOptions(
                    model_asset_path=str(MODEL_PATH)
                )
                opts = mp_vision.HandLandmarkerOptions(
                    base_options=base_opts,
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=1,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.detector = mp_vision.HandLandmarker.create_from_options(opts)
                self.mode = "tasks"
                print("  🤖 Hand engine: MediaPipe Tasks API (full accuracy)")
            except Exception as e:
                print(f"  ⚠ Tasks API init failed ({e}), using fallback")
                self.mode = "skin"
        else:
            self.mode = "skin"
            print("  ✋ Hand engine: Skin-color fallback mode")
            print(f"    (run python3 00_setup_model.py to enable full accuracy)")

        self._frame_ts = 0

    def process(self, bgr_frame):
        """
        Process one BGR frame.
        Returns: (landmarks_63d: np.ndarray | None, annotated_frame: np.ndarray)
          landmarks_63d — flat array of shape (63,): 21×(x,y,z) normalized
          annotated_frame — frame with skeleton drawn on it
        """
        if self.mode == "tasks":
            return self._process_tasks(bgr_frame)
        else:
            return self._process_skin(bgr_frame)

    # ── Tasks API path ────────────────────────────────────────────────────

    def _process_tasks(self, bgr_frame):
        self._frame_ts += 33          # ~30 fps fake timestamp
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(mp_image, self._frame_ts)

        annotated = bgr_frame.copy()
        if not result.hand_landmarks:
            return None, annotated

        hand = result.hand_landmarks[0]
        coords = []
        for lm in hand:
            coords.extend([lm.x, lm.y, lm.z])
        landmarks = np.array(coords, dtype=np.float32)
        normalized = _normalize(landmarks)

        # Draw skeleton
        h, w = bgr_frame.shape[:2]
        pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        _draw_skeleton(annotated, pts_px)

        return normalized, annotated

    # ── Skin-color fallback path ──────────────────────────────────────────

    def _process_skin(self, bgr_frame):
        annotated = bgr_frame.copy()
        h, w = bgr_frame.shape[:2]

        # Convert to YCrCb and HSV for skin detection
        ycrcb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
        hsv   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

        # Skin mask (YCrCb range)
        mask_y = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        # Skin mask (HSV range)
        mask_h = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
        mask   = cv2.bitwise_or(mask_y, mask_h)

        # Morphology clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, annotated

        # Take largest contour
        hand_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand_cnt) < 3000:
            return None, annotated

        # Draw contour outline
        cv2.drawContours(annotated, [hand_cnt], -1, (0, 220, 120), 2)

        # Estimate 21 pseudo-landmarks from contour + convex hull
        landmarks_px = _estimate_pseudo_landmarks(hand_cnt, (w, h))
        if landmarks_px is None:
            return None, annotated

        # Build 63-d vector (z=0 for all — not available from skin detection)
        coords = []
        for px, py in landmarks_px:
            coords.extend([px / w, py / h, 0.0])
        landmarks = np.array(coords, dtype=np.float32)
        normalized = _normalize(landmarks)

        # Draw pseudo-skeleton
        _draw_skeleton(annotated, landmarks_px)

        return normalized, annotated

    def close(self):
        if self.detector is not None:
            self.detector.close()


# ── Helpers ───────────────────────────────────────────────────────────────

def _normalize(landmarks: np.ndarray) -> np.ndarray:
    """Translate wrist to origin; scale by wrist→mid-MCP distance."""
    pts = landmarks.reshape(21, 3)
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts = pts / scale
    return pts.flatten().astype(np.float32)


def _draw_skeleton(img, pts_px: list, color=(0, 220, 120)):
    """Draw hand connections and joint dots."""
    for i, j in HAND_CONNECTIONS:
        if i < len(pts_px) and j < len(pts_px):
            cv2.line(img, pts_px[i], pts_px[j], color, 2, cv2.LINE_AA)
    for i, pt in enumerate(pts_px):
        dot_color = (0, 180, 255) if i == 0 else color
        cv2.circle(img, pt, 4, dot_color, -1, cv2.LINE_AA)


def _estimate_pseudo_landmarks(contour, frame_size):
    """
    Estimate 21 pseudo hand-landmarks from a skin contour.
    Uses: bounding box, centroid, convex hull defects, fingertip candidates.
    Not anatomically perfect but gives a consistent 63-d feature.
    """
    w, h = frame_size

    # Bounding rect
    x, y, bw, bh = cv2.boundingRect(contour)
    cx = x + bw // 2
    cy = y + bh // 2

    # Convex hull + defects for finger detection
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 4:
        return None
    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except Exception:
        return None

    # Fingertip candidates from convex hull
    hull_pts = cv2.convexHull(contour, returnPoints=True)
    hull_pts = hull_pts.reshape(-1, 2)

    # Sort hull points by y (top first = fingertips)
    top_pts = hull_pts[hull_pts[:, 1].argsort()][:8]

    # Estimate wrist: bottom-centre of bounding box
    wrist = (cx, y + bh)

    # Build 21 landmarks:
    # 0 = wrist
    # 1-4 = thumb approximation (linear from wrist to thumb side)
    # 5-8 = index
    # 9-12 = middle
    # 13-16 = ring
    # 17-20 = pinky
    #
    # Simple approach: divide contour points into 5 "finger zones"
    pts = contour.reshape(-1, 2)

    # Filter top region (upper 60% of bounding box = fingers area)
    top_mask = pts[:, 1] < y + bh * 0.6
    top_pts_all = pts[top_mask] if top_mask.sum() > 20 else pts

    # Divide into 5 horizontal zones
    zone_w = bw / 5
    fingers_tips = []
    for i in range(5):
        zx_min = x + i * zone_w
        zx_max = x + (i + 1) * zone_w
        zone_mask = (top_pts_all[:, 0] >= zx_min) & (top_pts_all[:, 0] < zx_max)
        zone_pts  = top_pts_all[zone_mask]
        if len(zone_pts) > 0:
            tip = zone_pts[zone_pts[:, 1].argmin()]  # highest point in zone
        else:
            tip = np.array([int(zx_min + zone_w/2), y])
        fingers_tips.append(tip)

    # Build 21-point skeleton
    landmarks = [tuple(map(int, wrist))]  # 0: wrist

    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    for fi in range(5):
        tip = fingers_tips[fi]
        # 3 intermediate points from wrist to tip
        for t in [0.3, 0.6, 0.85, 1.0]:
            px = int(wrist[0] + (tip[0] - wrist[0]) * t)
            py = int(wrist[1] + (tip[1] - wrist[1]) * t)
            landmarks.append((px, py))

    return landmarks[:21]  # guarantee exactly 21


# ── Standalone test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing HandEngine...")
    engine = HandEngine()
    cap = cv2.VideoCapture(0)
    print("Press Q to quit test.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        lm, ann = engine.process(frame)
        status = f"Landmarks: {lm is not None}"
        cv2.putText(ann, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 100), 2)
        cv2.putText(ann, f"Mode: {engine.mode}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("HandEngine Test", ann)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    engine.close()
    cv2.destroyAllWindows()
