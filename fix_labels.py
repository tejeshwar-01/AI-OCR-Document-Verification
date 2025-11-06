import os

def fix_label_file(label_path):
    new_lines = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            # Skip empty or invalid lines
            if not parts:
                continue
            # Polygon → convert to bbox
            if len(parts) > 5:
                try:
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x_center = (max(xs) + min(xs)) / 2
                    y_center = (max(ys) + min(ys)) / 2
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)
                    new_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                except:
                    continue
            elif len(parts) == 5:
                # already in YOLO format
                try:
                    cls = int(parts[0])
                    vals = list(map(float, parts[1:]))
                    if all(0 <= v <= 1 for v in vals):
                        new_lines.append(line.strip())
                except:
                    continue
            # ignore anything else
    # Save valid lines
    if new_lines:
        with open(label_path, "w") as f:
            f.write("\n".join(new_lines))
        return True
    else:
        os.remove(label_path)
        return False


root = r"C:\Users\tejes\OneDrive\Documents\Desktop\AI-OCR-Document-Verification-main\data\fraud_dataset"
total_fixed = 0
total_removed = 0

for folder, _, files in os.walk(root):
    for file in files:
        if file.endswith(".txt"):
            path = os.path.join(folder, file)
            ok = fix_label_file(path)
            if ok:
                total_fixed += 1
            else:
                total_removed += 1

print(f"✅ Fixed {total_fixed} label files, 🗑 Removed {total_removed} invalid ones.")
