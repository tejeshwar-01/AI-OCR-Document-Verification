"""
End-to-end runner:
steps:
1. data_collection (optional)
2. preprocess
3. ocr
4. verification
5. fraud detection (infer only, model must be trained separately)
"""
import os
from pathlib import Path
import subprocess
import argparse

def run_step(cmd_list):
    print("RUNNING:", " ".join(cmd_list))
    subprocess.check_call(cmd_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="./data/raw")
    parser.add_argument("--processed", default="./data/processed")
    parser.add_argument("--ocr_out", default="./output/ocr")
    parser.add_argument("--verif_out", default="./output/verification")
    parser.add_argument("--fraud_model", default="./models/yolo_fraud/fraud/weights/best.pt")
    parser.add_argument("--fraud_out", default="./output/fraud")
    parser.add_argument("--selfies", default=None)
    args = parser.parse_args()

    Path(args.processed).mkdir(parents=True, exist_ok=True)
    Path(args.ocr_out).mkdir(parents=True, exist_ok=True)
    Path(args.verif_out).mkdir(parents=True, exist_ok=True)
    Path(args.fraud_out).mkdir(parents=True, exist_ok=True)

    # 1 & 2 Preprocess
    run_step(["python", "preprocess.py", "--in_dir", args.raw, "--out_dir", args.processed])

    # 3 OCR
    run_step(["python", "ocr.py", "--processed", args.processed, "--out", args.ocr_out])

    # 4 Verification
    run_step(["python", "verifier.py", "--ocr", args.ocr_out, "--out", args.verif_out])

    # 5 Fraud infer
    run = ["python", "fraud_detector.py", "--model", args.fraud_model, "--processed", args.processed, "--verif", args.verif_out, "--out", args.fraud_out]
    if args.selfies:
        run += ["--selfies", args.selfies]
    run_step(run)

    print("PIPELINE COMPLETE. Check outputs in:", args.ocr_out, args.verif_out, args.fraud_out)
