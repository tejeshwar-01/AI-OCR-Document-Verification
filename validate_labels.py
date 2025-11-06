import os

root = "data/fraud_dataset"
removed = 0

for split in ["train","valid","test"]:
    lbl_dir = os.path.join(root, split, "labels")
    if not os.path.isdir(lbl_dir):
        continue
    for f in os.listdir(lbl_dir):
        if not f.endswith(".txt"):
            continue
        path = os.path.join(lbl_dir, f)
        with open(path) as file:
            lines = [l.strip() for l in file.readlines() if l.strip()]
        if not lines:
            os.remove(path)
            removed += 1
        else:
            valid = True
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    valid = False
                    break
            if not valid:
                os.remove(path)
                removed += 1

print(f"🧾 Total invalid label files removed: {removed}")
