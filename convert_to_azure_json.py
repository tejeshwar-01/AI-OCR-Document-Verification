import pandas as pd
import json
from pathlib import Path

# Paths
OUTPUT_DIR = Path(r"C:\Users\tejes\OneDrive\Documents\Desktop\AIOCR\processed_output")
CSV_PATH = OUTPUT_DIR / "processed_labels.csv"
OUTPUT_JSON = OUTPUT_DIR / "azure_formatted_data.json"

# Load processed labels
df = pd.read_csv(CSV_PATH)

# Detect possible image column
image_col = None
for possible_col in ["image_path", "path", "file", "filename", "image"]:
    if possible_col in df.columns:
        image_col = possible_col
        break

if not image_col:
    raise KeyError("No image path column found in CSV. Please check your processed_labels.csv headers.")

azure_data = []

for _, row in df.iterrows():
    img_path = row[image_col]

    label_data = []
    # Ensure coordinates exist
    if all(col in df.columns for col in ["x", "y", "w", "h"]):
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]

        # Convert YOLO (x_center, y_center, width, height) → bounding box corners
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y - h / 2
        x3, y3 = x + w / 2, y + h / 2
        x4, y4 = x - w / 2, y + h / 2

        label_data.append({
            "fieldName": str(row.get("class_name", "unknown_field")),
            "boundingBox": [x1, y1, x2, y2, x3, y3, x4, y4]
        })

    azure_data.append({
        "document": str(img_path),
        "labels": label_data
    })

# Save JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(azure_data, f, indent=4)

print(f"✅ Azure-compatible JSON saved to:\n{OUTPUT_JSON}")
