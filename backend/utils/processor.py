# ============================================================
#  processor.py  — UNIVERSAL AADHAAR OCR (Final Production Build)
# ============================================================

import os
import io
import re
import zipfile
import datetime
import traceback
import cv2
import numpy as np
from PIL import Image

# ---------- Project utilities ----------
from backend.utils.verification_rules import (
    validate_aadhaar_number,
    validate_name,
    validate_gender,
    validate_dob,
    correct_common_ocr_errors,
)

from backend.utils.ocr_utils import preprocess_for_ocr, preprocess_for_ocr_full


# ============================================================
#  Dependency Detection
# ============================================================

try:
    from ultralytics import YOLO
    _YOLO_PRESENT = True
except:
    _YOLO_PRESENT = False

try:
    import easyocr
    _EASY_PRESENT = True
except:
    _EASY_PRESENT = False

try:
    import pytesseract
    _TESS_PRESENT = True
except:
    _TESS_PRESENT = False

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    from pyaadhaar.utils import isSecureQr
    from pyaadhaar.decode import AadhaarSecureQr
    _QR_PRESENT = True
except:
    _QR_PRESENT = False


# ============================================================
#  SANITIZER
# ============================================================

def sanitize_value(v):
    if v is None:
        return None
    if isinstance(v, np.generic):
        try: return v.item()
        except: return str(v)
    if isinstance(v, np.ndarray):
        try: return v.tolist()
        except: return str(v)
    if isinstance(v, (bytes, bytearray)):
        try: return v.decode("utf-8", errors="ignore")
        except: return str(v)
    if isinstance(v, (list, tuple)):
        return [sanitize_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): sanitize_value(vv) for k, vv in v.items()}
    if isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return sanitize_value(obj)


# ============================================================
#  EASY OCR WRAPPERS
# ============================================================

_EASY_READER = None

def get_easyocr_reader():
    global _EASY_READER
    if not _EASY_PRESENT:
        return None
    if _EASY_READER is None:
        try:
            _EASY_READER = easyocr.Reader(["en"], gpu=False)
        except:
            _EASY_READER = easyocr.Reader(["en"], gpu=False)
    return _EASY_READER


def easyocr_image_to_text(pil_img):
    if not _EASY_PRESENT:
        return ""
    try:
        arr = np.array(pil_img.convert("RGB"))
        reader = get_easyocr_reader()
        results = reader.readtext(arr, detail=1)
        out = [r[1] for r in results]
        return " ".join(out).lower()
    except:
        return ""


# ============================================================
#  UNIVERSAL NAME EXTRACTOR (WORKS FOR ALL AADHAAR FORMATS)
# ============================================================

def extract_universal_name(full_pil):
    """
    Extracts name from:
    - Old laminated Aadhaar
    - New PVC Aadhaar
    - PDF-format Aadhaar
    Uses multiple region crops and chooses the best valid output.
    """
    W, H = full_pil.size

    regions = []

    # REGION 1 — Old Aadhaar name zone (right of photo, upper middle)
    regions.append(full_pil.crop((
        int(W * 0.28), int(H * 0.16),
        int(W * 0.95), int(H * 0.36)
    )))

    # REGION 2 — PVC Aadhaar (below photo)
    regions.append(full_pil.crop((
        int(W * 0.05), int(H * 0.28),
        int(W * 0.90), int(H * 0.48)
    )))

    # REGION 3 — High-res Aadhaar (top-left)
    regions.append(full_pil.crop((
        int(W * 0.05), int(H * 0.10),
        int(W * 0.70), int(H * 0.28)
    )))

    # REGION 4 — Generic fallback next to photo
    regions.append(full_pil.crop((
        int(W * 0.24), int(H * 0.18),
        int(W * 0.95), int(H * 0.50)
    )))

    candidates = []

    for crop in regions:
        try:
            txt = pytesseract.image_to_string(
                crop,
                config="--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz. "
            )
        except:
            txt = pytesseract.image_to_string(crop)

        clean = re.sub(r"[^A-Za-z .]", "", txt)
        clean = re.sub(r"\s{2,}", " ", clean).strip()

        # Accept only realistic names
        if (
            2 <= len(clean.split()) <= 5 and
            not any(bad in clean.lower() for bad in
                ["dob", "date", "male", "female", "government", "uidai"])
        ):
            candidates.append(clean)

    if candidates:
        # choose longest valid candidate
        return max(candidates, key=len)

    return ""


# ============================================================
#  STRICT DOB / GENDER OCR
# ============================================================

def ocr_dob_strict(pil_img):
    if not _TESS_PRESENT:
        return ""
    try:
        config = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/-."
        txt = pytesseract.image_to_string(pil_img, config=config)
        txt = txt.replace("-", "/").replace(".", "/")
        txt = re.sub(r"[^\d/]", "", txt)
        m = re.search(r"(\d{2})/(\d{2})/(\d{4})", txt)
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else txt
    except:
        return ""


def ocr_gender_strict(pil_img):
    try:
        if _EASY_PRESENT:
            t = easyocr_image_to_text(pil_img)
            if "male" in t: return "Male"
            if "female" in t: return "Female"
    except:
        pass
    try:
        txt = pytesseract.image_to_string(
            pil_img,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        ).lower()
        if "male" in txt: return "Male"
        if "female" in txt: return "Female"
    except:
        pass
    return ""


# ============================================================
#  Aadhaar Heuristic Check
# ============================================================

def is_aadhaar_image(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img)
        W, H = img.size

        aspect_ok = 1.1 <= (W/H) <= 2.9

        band = arr[:min(80, H//8)]
        avg = np.mean(band, axis=(0,1))
        orange = avg[0] > avg[1]*0.9 and avg[0] > avg[2]*0.85

        txt = ""
        if _EASY_PRESENT:
            txt = easyocr_image_to_text(img)
        elif _TESS_PRESENT:
            txt = pytesseract.image_to_string(img).lower()

        kw = sum(k in txt for k in ["aadhaar","uidai","dob","male","female"])
        numbers = len(re.findall(r"\d{4}\s?\d{4}\s?\d{4}", txt))

        score = 0
        score += 40 if aspect_ok else 0
        score += 25 if orange else 0
        score += kw * 10
        score += 25 if numbers else 0

        return score >= 45, min(score,100), {
            "aspect_ok": aspect_ok,
            "orange_band": orange,
            "keywords_found": kw,
            "numbers_found": numbers,
            "snippet": txt[:200]
        }
    except Exception as e:
        return False, 0, {"error": str(e)}


# ============================================================
#  TEXT HELPERS
# ============================================================

def find_aadhaar_in_text(txt):
    m = re.search(r"(\d{4}\s?\d{4}\s?\d{4})", txt or "")
    return m.group(1) if m else ""


def find_dob_in_text(txt):
    m = re.search(r"(\d{2})[/-](\d{2})[/-](\d{4})", txt or "")
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else ""


def guess_name_from_text(txt):
    if not txt:
        return ""
    lines = txt.splitlines()
    cands = []
    for ln in lines:
        ln = ln.strip()
        if (
            3 < len(ln) < 40 and
            ln.replace(" ", "").isalpha() and
            not any(w in ln.lower() for w in ["dob","male","female","government","uidai"])
        ):
            cands.append(ln)
    return max(cands, key=len) if cands else ""


# ============================================================
#  MAIN PROCESSOR
# ============================================================

def process_single_image_bytes(front_bytes, back_bytes=None, do_qr_check=False, device="cpu"):

    ts = datetime.datetime.now().isoformat()

    try:
        ok, conf, info = is_aadhaar_image(front_bytes)

        if not ok:
            return sanitize_json({
                "error": "NOT_AADHAAR",
                "message": "Image does not appear to be an Aadhaar card",
                "confidence_score": conf,
                "aadhaar_verification": info,
                "assessment": "INVALID_INPUT",
                "timestamp": ts
            })

        pil = Image.open(io.BytesIO(front_bytes)).convert("RGB")
        arr = np.array(pil)
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        results = {
            "timestamp": ts,
            "fraud_score": 0,
            "assessment": "LOW",
            "confidence_score": conf,
            "ocr_data": {},
            "indicators": [],
            "qr_data": {},
            "extracted": {},
            "aadhaar_verification": {
                "is_aadhaar_card": True,
                "confidence_score": conf,
                "verification_details": info
            }
        }

        # ---------------------------
        # Full OCR
        # ---------------------------
        if _EASY_PRESENT:
            full_text = easyocr_image_to_text(pil)
        else:
            full_text = pytesseract.image_to_string(pil).lower()

        results["ocr_data"]["full_text"] = full_text

        # ---------------------------
        # Aadhaar Number
        # ---------------------------
        aadhaar = find_aadhaar_in_text(full_text)
        aadhaar = re.sub(r"[^0-9]", "", aadhaar)

        # ---------------------------
        # DOB / Gender
        # ---------------------------
        dob = find_dob_in_text(full_text)
        gender = "Male" if "male" in full_text else ("Female" if "female" in full_text else "")

        # ---------------------------
        # UNIVERSAL NAME EXTRACTOR
        # ---------------------------
        name = extract_universal_name(pil)

        if not name:
            name = guess_name_from_text(full_text)

        name = re.sub(r"[^A-Za-z .]", "", name)
        name = re.sub(r"\s{2,}", " ", name).strip()

        # ---------------------------
        # VALIDATION
        # ---------------------------
        a_val = validate_aadhaar_number(aadhaar) or "Invalid"
        n_val = validate_name(name) or "Invalid"
        d_val = validate_dob(dob) or "Invalid"
        g_val = validate_gender(gender) or "Invalid"

        # Aadhaar
        if "Invalid" in a_val:
            results["fraud_score"] += 3
            results["indicators"].append("🔴 HIGH: Invalid Aadhaar number.")
        else:
            results["indicators"].append("✅ LOW: Aadhaar number valid.")

        # Name
        if "Invalid" in n_val:
            results["fraud_score"] += 1
            results["indicators"].append("🟡 MEDIUM: Name appears invalid.")
        else:
            results["indicators"].append("✅ LOW: Name valid.")

        # DOB
        if "Invalid" in d_val:
            results["fraud_score"] += 1
            results["indicators"].append("🟡 MEDIUM: DOB invalid.")
        else:
            results["indicators"].append("✅ LOW: DOB valid.")

        # Gender
        if "Invalid" in g_val:
            results["fraud_score"] += 1
            results["indicators"].append("🟡 MEDIUM: Gender invalid.")
        else:
            results["indicators"].append("✅ LOW: Gender valid.")

        # ---------------------------
        # Save extracted fields
        # ---------------------------
        results["extracted"] = {
            "name": name,
            "dob": dob,
            "gender": gender,
            "aadhaar": aadhaar
        }

        # ---------------------------
        # QR Decode (optional)
        # ---------------------------
        if do_qr_check and _QR_PRESENT:
            try:
                qr = decode_secure_qr(arr_bgr)
                results["qr_data"] = qr
            except:
                results["qr_data"] = {"error": "QR decoding failed"}

        # ---------------------------
        # Final Fraud Assessment
        # ---------------------------
        fs = results["fraud_score"]
        results["assessment"] = "LOW" if fs < 3 else ("MODERATE" if fs < 7 else "HIGH")

        return sanitize_json(results)

    except Exception as e:
        return sanitize_json({
            "error": "PROCESSING_FAILED",
            "message": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": ts
        })


# ============================================================
#  BATCH ZIP PROCESSOR
# ============================================================

def process_zip_bytes(zip_bytes, model_path=None, do_qr_check=False, device="cpu", max_files=None):

    out = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            imgs = [n for n in z.namelist() if n.lower().endswith((".jpg",".jpeg",".png"))]

            if max_files:
                imgs = imgs[:max_files]

            for name in imgs:
                try:
                    b = z.read(name)
                    result = process_single_image_bytes(b, None, do_qr_check, device)
                    result["filename"] = name
                    out.append(sanitize_json(result))
                except Exception as e:
                    out.append({"filename": name, "error": str(e)})
    except Exception as e:
        out.append({"error": str(e)})

    return out
