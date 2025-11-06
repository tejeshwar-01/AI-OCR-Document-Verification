from PIL import Image
import os

root = "data/fraud_dataset"
removed = []

for split in ["train", "valid", "test"]:
    folder = os.path.join(root, split, "images")
    if not os.path.isdir(folder):
        continue
    for f in os.listdir(folder):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder, f)
        try:
            Image.open(path).verify()
        except Exception:
            os.remove(path)
            removed.append(path)

print(f"🧾 Removed {len(removed)} corrupt images")
if removed:
    print("Example:", removed[:5])
