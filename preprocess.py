"""
Preprocess images:
- resize (max side)
- auto-rotate via EXIF
- adaptive threshold / denoise
- store metadata json per image
"""
import os
from pathlib import Path
from PIL import Image, ImageOps, ExifTags, ImageFilter
import json
import argparse

def load_image_fix_orientation(path):
    img = Image.open(path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            orient = exif.get(orientation, None)
            if orient == 3:
                img = img.rotate(180, expand=True)
            elif orient == 6:
                img = img.rotate(270, expand=True)
            elif orient == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img

def preprocess_image(img, max_side=1600):
    # auto-resize
    w,h = img.size
    scale = min(1.0, max_side / max(w,h))
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    # improve contrast / denoise
    img = img.convert("RGB")
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img

def run(in_dir, out_dir):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for p in in_dir.glob("*"):
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            continue
        img = load_image_fix_orientation(p)
        img = preprocess_image(img)
        out_path = out_dir / p.name
        img.save(out_path, quality=90)
        m = {
            "filename": p.name,
            "orig_path": str(p),
            "processed_path": str(out_path),
            "size": img.size
        }
        meta.append(m)
    # write metadata
    meta_file = out_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)
    print(f"Processed {len(meta)} images -> {out_dir}")
    return meta_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="./data/raw")
    parser.add_argument("--out_dir", default="./data/processed")
    args = parser.parse_args()
    run(args.in_dir, args.out_dir)
