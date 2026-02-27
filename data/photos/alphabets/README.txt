USL PHOTO DATASET — ALPHABETS
===========================================
Add hand-sign photos to each subfolder.

Folder structure:
  data/photos/alphabets/A/photo1.jpg
  data/photos/alphabets/A/photo2.jpg
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
  python3 collect_from_photos.py --mode alphabet
  python3 02_eda_analysis.py --mode alphabet
  python3 03_train_models.py --mode alphabet
