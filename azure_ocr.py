# azure_ocr.py
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Azure SDK
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient   # azure-ai-formrecognizer / documentintelligence client

load_dotenv()

ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")

INPUT_DIR = Path(r"C:\Users\tejes\OneDrive\Documents\Desktop\AIOCR\archive")
OUTPUT_DIR = Path(r"C:\Users\tejes\OneDrive\Documents\Desktop\AIOCR\processed_output\azure_ocr_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = DocumentAnalysisClient(ENDPOINT, AzureKeyCredential(KEY))

# choose model: identity document model recommended for Aadhaar (prebuilt:idDocument)
MODEL_ID = "prebuilt-idDocument"   # or "prebuilt-document" / custom model if you trained one

def analyze_image(image_path: Path):
    with open(image_path, "rb") as f:
        poller = client.begin_analyze_document(MODEL_ID, document=f)
        result = poller.result()
    return result

def extract_result_to_dict(result):
    # Convert the SDK result to a JSON-friendly structure
    out = {"fields": [], "raw": []}
    # fields: id documents/artifact extraction depends on model; this generic code pulls key values
    for kv in getattr(result, "key_value_pairs", []) or []:
        k = getattr(kv, "key", None)
        v = getattr(kv, "value", None)
        out["fields"].append({
            "key_text": k.content if k else None,
            "value_text": v.content if v else None,
            "key_bbox": getattr(k, "bounding_regions", None),
            "value_bbox": getattr(v, "bounding_regions", None)
        })
    # fallback: text content
    out["raw_text"] = ""
    for page in result.pages:
        out["raw_text"] += " ".join([word.content for line in page.lines for word in line.bounding_regions or []]) if hasattr(page, "lines") else ""
    return out

def main():
    # Walk your processed images folder or archive images
    images = list((INPUT_DIR / "train" / "images").glob("*.jpg")) + \
             list((INPUT_DIR / "valid" / "images").glob("*.jpg")) + \
             list((INPUT_DIR / "test" / "images").glob("*.jpg"))

    summary = []
    for img in images:
        print("Analyzing:", img)
        try:
            result = analyze_image(img)
            # Save raw JSON result
            out_path = OUTPUT_DIR / (img.stem + ".json")
            with open(out_path, "w", encoding="utf-8") as f:
                # many SDK objects are not serializable: convert simple structure
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            # build light summary row
            summary.append({"image": str(img), "json": str(out_path)})
        except Exception as e:
            print("ERROR analyzing", img, e)
            summary.append({"image": str(img), "json": None, "error": str(e)})

    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "ocr_summary.csv", index=False)
    print("Done. Saved OCR results to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
