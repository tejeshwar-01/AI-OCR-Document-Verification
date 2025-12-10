# backend/utils/processor.py
import os
import io
import re
import zipfile
import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw

# 🚨 FORCE DISABLE YOLO FOR RENDER FREE TIER
YOLO_AVAILABLE = False

# -------------------------------------------------------
# IMPORT RULES / OCR HELPERS
# -------------------------------------------------------
from backend.utils.verification_rules import (
    validate_aadhaar_number,
    validate_name,
    validate_gender,
    validate_dob,
    correct_common_ocr_errors,
)

from backend.utils.ocr_utils import preprocess_for_ocr

# -------------------------------------------------------
# OPTIONAL DEPENDENCIES
# -------------------------------------------------------
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    from pyaadhaar.utils import isSecureQr
    from pyaadhaar.decode import AadhaarSecureQr
    PYAADHAAR_AVAILABLE = True
except Exception:
    PYAADHAAR_AVAILABLE = False

# -------------------------------------------------------
# EASY OCR INITIALIZATION
# -------------------------------------------------------
_EASYREADER = None
def get_easyocr_reader():
    global _EASYREADER
    if not EASYOCR_AVAILABLE:
        return None
    if _EASYREADER is None:
        try:
            _EASYREADER = easyocr.Reader(["en"], gpu=False)
        except:
            _EASYREADER = None
    return _EASYREADER


# -------------------------------------------------------
# YOLO LOAD (DISABLED)
# -------------------------------------------------------
def load_models(device="cpu"):
    """Disabled on Render Free Tier"""
    return


CUSTOM_MODEL = None
FACE_MODEL = None


# -------------------------------------------------------
# OCR — FULL IMAGE
# -------------------------------------------------------
def easyocr_image_to_text(pil_image):
    if not EASYOCR_AVAILABLE:
        return ""
    try:
        reader = get_easyocr_reader()
        if reader is None:
            return ""
        arr = np.array(pil_image.convert("RGB"))
        results = reader.readtext(arr, detail=1)
        text_list = [r[1] for r in results]
        return " ".join(text_list).lower()
    except:
        return ""


# -------------------------------------------------------
# OCR — CROP
# -------------------------------------------------------
def easyocr_crop_to_text(crop_pil):
    if not EASYOCR_AVAILABLE:
        return ""
    try:
        reader = get_easyocr_reader()
        if reader is None:
            return ""
        arr = np.array(crop_pil.convert("RGB"))
        results = reader.readtext(arr, detail=0)
        return " ".join(results) if isinstance(results, list) else str(results)
    except:
        return ""


# -------------------------------------------------------
# QR DECODING
# -------------------------------------------------------
def decode_secure_qr(image_np_bgr):
    if not PYAADHAAR_AVAILABLE:
        return {"error": "QR decoding disabled"}
    try:
        gray = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2GRAY)
        codes = pyzbar_decode(gray)
        if not codes:
            return {"error": "No QR detected"}
        raw = codes[0].data
        if isSecureQr(raw):
            secure = AadhaarSecureQr(int(raw))
            data = secure.decodeddata()
            return dict(data)
        return {"error": "Not a secure Aadhaar QR"}
    except Exception as e:
        return {"error": f"QR error: {str(e)}"}


# -------------------------------------------------------
# FIXED — Aadhaar Image Heuristic
# -------------------------------------------------------
def is_aadhaar_image(image_bytes):
    """
    Improved Aadhaar detection:
    - Does NOT depend on OCR
    - Uses aspect ratio + size + saffron band + OCR fallback
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        w, h = image.size
        aspect = w / h if h else 0
        aspect_ok = 1.1 <= aspect <= 2.9
        size_ok = min(w, h) >= 160

        # detect saffron color band top 80px
        npimg = np.array(image)
        top = npimg[:80, :, :]
        avg = np.mean(top, axis=(0, 1))
        orange = avg[0] > avg[1] > avg[2] * 0.7

        # OCR optional
        keywords = 0
        aadhaar_numbers = 0
        text_snippet = ""

        if EASYOCR_AVAILABLE:
            text = easyocr_image_to_text(image)
            text_snippet = text[:250]

            KEY_LIST = ["aadhaar", "aadhar", "uidai", "government", "dob", "male", "female"]
            keywords = sum(1 for k in KEY_LIST if k in text)
            aadhaar_numbers = len(re.findall(r"\b\d{4}\s?\d{4}\s?\d{4}\b", text))

        # Score
        score = 0
        if aspect_ok: score += 35
        if size_ok:   score += 20
        if orange:    score += 20
        score += min(keywords * 15, 30)
        score += 25 if aadhaar_numbers else 0
        score = min(score, 100)

        return score >= 40, score, {
            "aspect_ratio_valid": aspect_ok,
            "size_valid": size_ok,
            "color_band_detected": orange,
            "keywords_found": keywords,
            "aadhaar_numbers_found": aadhaar_numbers,
            "detected_text_snippet": text_snippet,
        }

    except Exception as e:
        return False, 0, {"error": str(e)}


# -------------------------------------------------------
# MAIN — SINGLE IMAGE PROCESSING
# -------------------------------------------------------
def process_single_image_bytes(front_bytes, back_bytes=None, do_qr_check=False, device="cpu"):
    ts = datetime.datetime.now().isoformat()

    # STEP 1 — Aadhaar detection
    is_aadhar, conf, details = is_aadhaar_image(front_bytes)

    if not is_aadhar:
        return {
            "error": "NOT_AADHAAR",
            "message": "Image does not appear to be Aadhaar",
            "confidence_score": conf,
            "aadhaar_verification": details,
            "timestamp": ts,
            "assessment": "INVALID_INPUT"
        }

    # STEP 2 — OCR fallback
    pil_img = Image.open(io.BytesIO(front_bytes)).convert("RGB")
    np_img = np.array(pil_img)

    results = {
        "timestamp": ts,
        "fraud_score": 0,
        "assessment": "LOW",
        "confidence_score": conf,
        "qr_data": {},
        "indicators": [],
        "ocr_data": {},
        "extracted": {},
        "aadhaar_verification": {
            "is_aadhaar_card": True,
            "confidence_score": conf,
            "verification_details": details
        }
    }

    # FULL OCR
    if EASYOCR_AVAILABLE:
        full = easyocr_image_to_text(pil_img)
        results["ocr_data"]["full_text"] = full
    else:
        full = ""

    # FIELD EXTRACTION
    name = ""
    dob = ""
    gender = ""
    aadhaar = ""

    # aadhaar number
    m = re.search(r"\b(\d{4}\s?\d{4}\s?\d{4})\b", full)
    if m:
        aadhaar = m.group(1).replace(" ", "")

    # gender
    gtext = full.lower()
    if "male" in gtext:
        gender = "Male"
    elif "female" in gtext:
        gender = "Female"

    # dob
    d = re.search(r"\d{2}/\d{2}/\d{4}", full)
    if d:
        dob = d.group(0)

    # name heuristic
    lines = [l.strip() for l in full.splitlines() if l.strip()]
    for ln in lines[:10]:
        if ln.isupper() and len(ln) > 3:
            name = ln
            break

    results["extracted"] = {
        "name": name,
        "dob": dob,
        "gender": gender,
        "aadhaar": aadhaar,
    }

    # VALIDATION
    if not aadhaar:
        results["fraud_score"] += 3
        results["indicators"].append("Missing Aadhaar number")
    else:
        status = validate_aadhaar_number(aadhaar)
        if "Invalid" in status:
            results["fraud_score"] += 3
            results["indicators"].append("Invalid Aadhaar number")
        else:
            results["indicators"].append("Aadhaar number valid")

    # name
    if not name:
        results["fraud_score"] += 1

    # dob
    if dob:
        if "Invalid" in validate_dob(dob):
            results["fraud_score"] += 2

    # gender missing
    if not gender:
        results["fraud_score"] += 1

    # QR CHECK
    if do_qr_check:
        try:
            qr = decode_secure_qr(cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
            results["qr_data"] = qr
        except:
            results["qr_data"] = {"error": "QR decode failed"}

    # FINAL RISK
    fs = results["fraud_score"]
    if fs >= 7: results["assessment"] = "HIGH"
    elif fs >= 3: results["assessment"] = "MODERATE"

    return results


# -------------------------------------------------------
# BATCH ZIP PROCESSING
# -------------------------------------------------------
def process_zip_bytes(zip_bytes, do_qr_check=False, device="cpu", max_files=None):
    out = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            imgs = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]

            if max_files:
                imgs = imgs[:max_files]

            for name in imgs:
                try:
                    data = z.read(name)
                    r = process_single_image_bytes(data, None, do_qr_check, device)
                    r["filename"] = name
                    out.append(r)
                except Exception as e:
                    out.append({"filename": name, "error": str(e)})

    except Exception as e:
        out.append({"error": str(e)})

    return out
