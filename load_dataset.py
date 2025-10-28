import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm

# --- Define paths ---
BASE_PATH = Path(r"C:\Users\tejes\OneDrive\Documents\Desktop\AIOCR")
OUTPUT_PATH = BASE_PATH / "processed_output"

# Ensure the folder exists
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# --- Load images ---
image_data = []
image_names = []
splits = ["train", "valid", "test"]

print("Loading and processing images...")

for split in splits:
    image_folder = BASE_PATH / "archive" / split / "images"
    if not image_folder.exists():
        print(f"⚠️ Skipping missing folder: {image_folder}")
        continue

    for img_path in tqdm(list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_resized = cv2.resize(img, (224, 224))  # resize for consistency
        image_data.append(img_resized)
        image_names.append((img_path.name, split))

# --- Save processed data ---
images_array = np.array(image_data)
index_df = pd.DataFrame(image_names, columns=["image", "split"])

np.save(OUTPUT_PATH / "processed_images.npy", images_array)
index_df.to_csv(OUTPUT_PATH / "processed_index.csv", index=False)

print("✅ Images and index saved in:", OUTPUT_PATH)
