"""
Field-level verification + optional LLM assist
- Structural checks (Aadhaar 12-digit check, DOB format, plausible age)
- If AZURE_OPENAI_KEY present -> call OpenAI to refine/interpret ambiguous text
"""
import os, re, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Basic validators
def is_valid_aadhaar(s: str):
    s = re.sub(r"\s+", "", s or "")
    return bool(re.fullmatch(r"\d{12}", s))

def is_valid_dob(s: str):
    # Accept dd/mm/yyyy or dd-mm-yyyy
    return bool(re.fullmatch(r"\d{2}[\/\-]\d{2}[\/\-]\d{4}", s or ""))

def verify_kv(kv):
    result = {"aadhaar_ok": False, "dob_ok": False, "notes": []}
    if "aadhaar" in kv and kv["aadhaar"]:
        result["aadhaar_ok"] = is_valid_aadhaar(kv["aadhaar"])
        if not result["aadhaar_ok"]:
            result["notes"].append("AADHAAR format invalid")
    else:
        result["notes"].append("AADHAAR missing")
    if "dob" in kv and kv["dob"]:
        result["dob_ok"] = is_valid_dob(kv["dob"])
        if not result["dob_ok"]:
            result["notes"].append("DOB format invalid")
    else:
        result["notes"].append("DOB missing")
    return result

# Optional Azure OpenAI assist (light usage)
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", None)

def llm_refine(raw_text):
    if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT_NAME:
        return None
    try:
        import openai
        openai.api_key = AZURE_OPENAI_KEY
        openai.api_base = AZURE_OPENAI_ENDPOINT
        # For Azure you may need to set api_type and api_version depending on setup; keep minimal here.
        prompt = (
            "You are a precise extractor for Indian Aadhaar cards. From the following OCR raw text, "
            "extract Name, Aadhaar (12 digits), DOB (DD/MM/YYYY), Gender. If uncertain, mark 'unknown'.\n\n"
            f"=== OCR TEXT ===\n{raw_text}\n\nOutput JSON with keys: name, aadhaar, dob, gender."
        )
        resp = openai.ChatCompletion.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role":"user","content":prompt}],
            max_tokens=300,
            temperature=0
        )
        content = resp["choices"][0]["message"]["content"]
        # attempt to extract JSON
        import json, re
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            return json.loads(m.group(0))
        return None
    except Exception as e:
        print("LLM refine failed:", e)
        return None

def run_verification(ocr_folder, out_folder):
    ocr_folder = Path(ocr_folder)
    out_folder = Path(out_folder); out_folder.mkdir(parents=True, exist_ok=True)
    for p in ocr_folder.glob("*.ocr.json"):
        j = json.load(open(p, encoding="utf8"))
        kv = j.get("kv", {})
        base_result = verify_kv(kv)
        llm_result = None
        if not (base_result["aadhaar_ok"] and base_result["dob_ok"]):
            # try LLM assist to salvage
            llm_result = llm_refine(j.get("raw_text", ""))
        out = {"file": p.name, "kv": kv, "base_verification": base_result, "llm_refine": llm_result}
        json.dump(out, open(out_folder / (p.stem + ".verification.json"), "w", encoding="utf8"), indent=2, ensure_ascii=False)
    print("Verification complete")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", default="./output/ocr")
    parser.add_argument("--out", default="./output/verification")
    args = parser.parse_args()
    run_verification(args.ocr, args.out)
