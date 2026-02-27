"""
=============================================================
USL — CAMERA DIAGNOSTIC & FIX
=============================================================
Run this to find your working camera and fix the timeout.

  python3 diagnose_camera.py
=============================================================
"""
import subprocess, sys, pathlib, os

print("=" * 55)
print("  USL Camera Diagnostic")
print("=" * 55)

# 1. List all /dev/video* devices
print("\n[1] Video devices found:")
videos = sorted(pathlib.Path("/dev").glob("video*"))
if not videos:
    print("    ❌ No /dev/video* devices found at all!")
    print("    → Your webcam may not be connected or recognised.")
else:
    for v in videos:
        print(f"    {v}")

# 2. Check if user is in 'video' group
import grp, pwd
username = os.environ.get("USER", "")
try:
    video_group = grp.getgrnam("video")
    members = video_group.gr_mem
    if username in members:
        print(f"\n[2] ✅ User '{username}' is in the 'video' group")
    else:
        print(f"\n[2] ⚠  User '{username}' is NOT in the 'video' group")
        print(f"    Fix: sudo usermod -aG video {username}")
        print(f"    Then log out and back in.")
except KeyError:
    print("\n[2] 'video' group not found (unusual)")

# 3. Try each camera index with OpenCV
print("\n[3] Testing camera indices with OpenCV...")
try:
    import cv2
    working = []
    for idx in range(5):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print(f"    ✅ index {idx} — WORKS  (frame shape: {frame.shape})")
                working.append(idx)
            else:
                print(f"    ⚠  index {idx} — opens but no frame (timeout)")
        else:
            print(f"    ✗  index {idx} — cannot open")

    if working:
        print(f"\n    ✅ Working camera index: {working[0]}")
        print(f"    Use: python3 01_collect_dataset.py --camera {working[0]}")
        print(f"    Use: python3 04_app.py --camera {working[0]}")
    else:
        print("\n    ❌ No working camera found with OpenCV")
except ImportError:
    print("    OpenCV not available in this environment")

# 4. Check v4l2 info
print("\n[4] Checking v4l2 device info...")
try:
    result = subprocess.run(["v4l2-ctl", "--list-devices"],
                            capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("    v4l2-ctl not found — install with: sudo apt install v4l-utils")
except (FileNotFoundError, subprocess.TimeoutExpired):
    print("    v4l2-ctl not available")

# 5. Check if running in VM / Wayland
print("\n[5] Environment check:")
session = os.environ.get("XDG_SESSION_TYPE", "unknown")
display = os.environ.get("DISPLAY", "not set")
wayland = os.environ.get("WAYLAND_DISPLAY", "not set")
print(f"    Session type : {session}")
print(f"    DISPLAY      : {display}")
print(f"    WAYLAND      : {wayland}")

if session == "wayland":
    print("\n    ⚠  You're on WAYLAND. OpenCV/Qt may have display issues.")
    print("    Fix: add this before running scripts:")
    print("      export QT_QPA_PLATFORM=xcb")
    print("    OR run with:")
    print("      QT_QPA_PLATFORM=xcb python3 04_app.py --mode alphabet")

# 6. Check if USB camera is physically present (lsusb)
print("\n[6] USB devices (looking for cameras):")
try:
    result = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=5)
    lines = result.stdout.strip().split("\n")
    cam_keywords = ["camera", "webcam", "video", "uvc", "logitech", "microsoft",
                    "trust", "creative", "sonix", "realtek", "genesys", "chicony"]
    found_cams = [l for l in lines if any(k in l.lower() for k in cam_keywords)]
    if found_cams:
        for l in found_cams:
            print(f"    📷 {l}")
    else:
        print("    No obvious camera USB devices found")
        print("    All USB devices:")
        for l in lines[:10]:
            print(f"      {l}")
except FileNotFoundError:
    print("    lsusb not available")

# 7. Try alternative backends
print("\n[7] Testing alternative OpenCV backends...")
try:
    import cv2
    backends = [
        (cv2.CAP_V4L2,  "V4L2"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (cv2.CAP_FFMPEG, "FFmpeg"),
    ]
    for backend, name in backends:
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                status = f"✅ frame={frame.shape}" if (ret and frame is not None) else "⚠ opens, no frame"
            else:
                status = "✗ cannot open"
            print(f"    {name:12}: {status}")
        except Exception as e:
            print(f"    {name:12}: error — {e}")
except ImportError:
    pass

print("\n" + "=" * 55)
print("  SUMMARY & NEXT STEPS")
print("=" * 55)
print("""
Most common causes of 'select() timeout':
  1. No webcam connected            → Connect a USB camera
  2. Wrong camera index             → Try --camera 1 or --camera 2
  3. Camera used by another app     → Close Cheese, Firefox, etc.
  4. Missing permissions            → sudo usermod -aG video $USER
  5. Wayland display issues         → Use: QT_QPA_PLATFORM=xcb python3 ...
  6. Running in a VM                → Enable USB passthrough in VM settings

Quick test commands:
  # Test camera directly:
  python3 -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened(), cap.read()[0]); cap.release()"

  # Run with Wayland fix:
  QT_QPA_PLATFORM=xcb python3 04_app.py --mode alphabet

  # Try camera index 1:
  python3 01_collect_dataset.py --mode alphabet --camera 1
""")
