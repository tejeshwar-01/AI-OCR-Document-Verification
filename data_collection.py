
import os
import shutil
from pathlib import Path
import argparse

def collect_from_folder(src_folder, dest_folder):
    src = Path(src_folder)
    dst = Path(dest_folder)
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in src.glob("*"):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            shutil.copy2(p, dst / p.name)
            count += 1
    print(f"Copied {count} image files from {src} to {dst}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="./incoming", help="source folder")
    parser.add_argument("--dst", default="./data/raw", help="destination data folder")
    args = parser.parse_args()
    collect_from_folder(args.src, args.dst)
