"""
OCR wrapper:
- If OCR_BACKEND=azure -> use Azure Document Intelligence (Form Recognizer)
- else -> use pytesseract

Outputs a JSON per image with extracted text fields and raw OCR text.
"""
import os
import re
from pathlib import Path
import json
from dotenv import load_dotenv
load_dotenv()

OCR_BACKEND = os.getenv("OCR_BACKEND", "tesseract").lower()

# common helpers
def save_json(target_path, data):
    with open(target_path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if OCR_BACKEND == "azure":
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.formrecognizer import DocumentAnalysisClient
    ENDPOINT = os.getenv("AZURE_FORM_ENDPOINT")
    KEY = os.getenv("AZURE_FORM_KEY")
    MODEL_ID = os.getenv("AZURE_FORM_MODEL_ID", "prebuilt-document")
    client = DocumentAnalysisClient(ENDPOINT, AzureKeyCredential(KEY))

    def azure_ocr(image_path):
        with open(image_path, "rb") as f:
            poller = client.begin_analyze_document(MODEL_ID, f)
            result = poller.result()
        # collect raw text and key-value pairs
        text = ""
        kv = {}
        for p in result.pages:
            text += " ".join([sp.content for sp in p.paragraphs]) + "\n"
        if result.key_value_pairs:
            for kvp in result.key_value_pairs:
                k = kvp.key.content if kvp.key else ""
                v = kvp.value.content if kvp.value else ""
                kv[k] = v
        return {"raw_text": text, "kv": kv}

    def run_ocr_on_folder(processed_dir, out_dir):
        processed_dir = Path(processed_dir)
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        for p in processed_dir.glob("*"):
            if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
                continue
            r = azure_ocr(str(p))
            save_json(out_dir / (p.stem + ".ocr.json"), r)
        print("Azure OCR finished")

else:
    # pytesseract fallback
    import pytesseract
    from PIL import Image
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", None)
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    def tesseract_ocr(image_path):
        img = Image.open(image_path)
        raw = pytesseract.image_to_string(img, lang="eng")
        # rough heuristic extractions for Aadhaar: 12-digit number, DOB, Name
        aadhaar_re = re.compile(r"\b(\d{4}\s?\d{4}\s?\d{4})\b")
        dob_re = re.compile(r"\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b")
        aad = aadhaar_re.search(raw)
        dob = dob_re.search(raw)
        kv = {}
        if aad:
            kv["aadhaar"] = aad.group(1).replace(" ", "")
        if dob:
            kv["dob"] = dob.group(1)
        return {"raw_text": raw, "kv": kv}

    def run_ocr_on_folder(processed_dir, out_dir):
        processed_dir = Path(processed_dir)
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        for p in processed_dir.glob("*"):
            if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
                continue
            r = tesseract_ocr(str(p))
            save_json(out_dir / (p.stem + ".ocr.json"), r)
        print("Tesseract OCR finished")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="./data/processed")
    parser.add_argument("--out", default="./output/ocr")
    args = parser.parse_args()
    run_ocr_on_folder(args.processed, args.out)
