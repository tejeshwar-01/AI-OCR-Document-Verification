import numpy as np
import pandas as pd
from pathlib import Path

# --- Paths ---
BASE_PATH = Path(r"C:\Users\tejes\OneDrive\Documents\Desktop\AIOCR")
OUTPUT_PATH = BASE_PATH / "processed_output"

IMAGES_PATH = OUTPUT_PATH / "processed_images.npy"
INDEX_PATH = OUTPUT_PATH / "processed_index.csv"
FINAL_LABELS_PATH = OUTPUT_PATH / "processed_labels.csv"

# --- Load processed data ---
images = np.load(IMAGES_PATH)
index_df = pd.read_csv(INDEX_PATH)

label_data = []

print("Matching YOLO labels with images...")

for _, row in index_df.iterrows():
    image_name = row["image"]
    split = row["split"]
    label_file = BASE_PATH / "archive" / split / "labels" / (Path(image_name).stem + ".txt")

    if label_file.exists():
        with open(label_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # skip invalid lines

                cls = float(parts[0])
                x, y, w, h = map(float, parts[1:5])
                extras = parts[5:] if len(parts) > 5 else []

                label_data.append({
                    "image": image_name,
                    "split": split,
                    "class": int(cls),
                    "x_center": x,
                    "y_center": y,
                    "width": w,
                    "height": h,
                    "extra_values": " ".join(extras) if extras else None
                })
    else:
        label_data.append({
            "image": image_name,
            "split": split,
            "class": None,
            "x_center": None,
            "y_center": None,
            "width": None,
            "height": None,
            "extra_values": None
        })

# --- Save labels ---
labels_df = pd.DataFrame(label_data)
labels_df.to_csv(FINAL_LABELS_PATH, index=False)

print(f"✅ Labels matched and saved to: {FINAL_LABELS_PATH}")
