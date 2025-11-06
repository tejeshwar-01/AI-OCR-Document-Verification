"""
Fraud detection:
- Uses Ultralytics YOLO (user supplies labeled dataset) to detect physical tampering:
  e.g., overlays, cut-and-paste, stamps, suspicious bounding boxes.
- Also uses face_recognition to compare face in document vs selfie (if selfie provided).
This file contains:
- train_yolo(): wrapper to train a YOLO model (assumes /data/fraud_dataset in YOLO format)
- infer_and_flag(): loads a model and runs inference across processed images; outputs risk score
"""
import os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from ultralytics import YOLO
import face_recognition
from PIL import Image
import numpy as np

def train_yolo(dataset_yaml, epochs=50, output_dir="./models/yolo_fraud"):
    """
    dataset_yaml: path to YAML describing dataset in YOLO format (train/val)
    """
    model = YOLO("yolov8n.pt")  # small base model
    model.train(data=str(dataset_yaml), epochs=epochs, project=output_dir, name="fraud")
    print("Training finished. Models placed in:", output_dir)

def face_match_score(doc_image_path, selfie_image_path):
    try:
        doc_img = face_recognition.load_image_file(doc_image_path)
        selfie_img = face_recognition.load_image_file(selfie_image_path)
        doc_faces = face_recognition.face_encodings(doc_img)
        selfie_faces = face_recognition.face_encodings(selfie_img)
        if not doc_faces or not selfie_faces:
            return 0.0  # no face found on one of the images -> suspicious
        d = face_recognition.face_distance([doc_faces[0]], selfie_faces[0])[0]
        # lower distance = more similar. convert to similarity score 0..1
        score = max(0.0, 1.0 - d)
        return float(score)
    except Exception as e:
        print("face match error:", e)
        return 0.0

def infer_and_flag(model_path, processed_dir, ocr_verif_dir, selfies_dir=None, out_dir="./output/fraud"):
    processed_dir = Path(processed_dir)
    ocr_verif_dir = Path(ocr_verif_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    summary = []
    for p in processed_dir.glob("*"):
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            continue
        res = model.predict(str(p), imgsz=1024, conf=0.3, verbose=False)
        boxes = []
        for r in res:
            # each r.boxes contains boxes with cls and conf
            try:
                for b in r.boxes:
                    boxes.append({"xyxy": b.xyxy.tolist(), "conf": float(b.conf[0]), "cls": int(b.cls[0])})
            except Exception:
                pass
        # basic heuristic risk score: #detections * avg_conf
        avg_conf = float(np.mean([b["conf"] for b in boxes])) if boxes else 0.0
        num = len(boxes)
        risk = min(1.0, num * avg_conf)
        # face match if selfie exists
        face_score = None
        selfie_path = None
        if selfies_dir:
            candidate = Path(selfies_dir) / p.name
            if candidate.exists():
                face_score = face_match_score(str(p), str(candidate))
                # degrade risk if face low
                if face_score < 0.4:
                    risk = max(risk, 0.8)
                selfie_path = str(candidate)
        # combine with verification flags (if verification shows invalid Aadhaar)
        verif_file = ocr_verif_dir / (p.stem + ".verification.json")
        verif = {}
        if verif_file.exists():
            verif = json.load(open(verif_file, encoding="utf8"))
            base_ok = verif.get("base_verification", {})
            if not base_ok.get("aadhaar_ok", False):
                risk = min(1.0, risk + 0.3)
        out = {
            "file": p.name,
            "detections": boxes,
            "num_detections": num,
            "avg_conf": float(avg_conf),
            "initial_risk": float(risk),
            "face_score": face_score,
            "selfie_path": selfie_path,
            "verification": verif
        }
        json.dump(out, open(out_dir / (p.stem + ".fraud.json"), "w", encoding="utf8"), indent=2, ensure_ascii=False)
        summary.append({"file": p.name, "risk": out["initial_risk"], "num_det": num})
    # write summary
    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf8"), indent=2)
    print("Fraud inference finished. Results ->", out_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to yolo model .pt")
    parser.add_argument("--processed", default="./data/processed")
    parser.add_argument("--verif", default="./output/verification")
    parser.add_argument("--selfies", default=None, help="optional folder with selfie images matching filenames")
    parser.add_argument("--out", default="./output/fraud")
    args = parser.parse_args()
    infer_and_flag(args.model, args.processed, args.verif, selfies_dir=args.selfies, out_dir=args.out)
