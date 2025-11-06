# AI-OCR-Document-Verification
An AI-powered Optical Character Recognition (OCR) and document verification system using Azure Document Intelligence and Azure OpenAI. This project automatically extracts and validates data from scanned Aadhaar cards and other ID documents.

1. Install dependencies (prefer a venv)
   pip install -r requirements.txt

2. Configure .env (copy .env.example -> .env) and set TESSERACT_CMD (or set OCR_BACKEND=azure and add keys).

3. Drop Aadhaar images into ./data/raw (filenames preserved). If you have selfie images, name them the same as the doc image and place in ./data/selfies.

4. Train YOLO fraud model (optional) or use pre-trained detection model:
   - Provide dataset in YOLO format and call train_yolo() (or run: python -c "from fraud_detector import train_yolo; train_yolo('data/fraud_dataset.yaml', epochs=50)")

5. Run pipeline:
   python pipeline_run.py --raw ./data/raw --processed ./data/processed --ocr_out ./output/ocr --verif_out ./output/verification --fraud_model ./models/yolo_fraud/fraud/weights/best.pt --fraud_out ./output/fraud --selfies ./data/selfies

Outputs:
 - ./output/ocr/*.ocr.json
 - ./output/verification/*.verification.json
 - ./output/fraud/*.fraud.json
 - ./output/fraud/summary.json
