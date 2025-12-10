# backend/utils/processor.py
"""
Processor utilities - YOLO DISABLED build (safe for Render Free)
- YOLO loading disabled to avoid OOM & worker-kill on small instances.
- EasyOCR and QR decoding are optional (used only if dependencies present).
- All outputs are sanitized to be JSON serializable.
"""

import os
import io
import re
import zipfile
import datetime
import traceback
import cv2
import numpy as np
from PIL import Image

# Force YOLO off for Render Free stability
YOLO_AVAILABLE = False

# Project utils (assume these exist and are correct)
from backend.utils.verification_rules import (
    validate_aadhaar_number,
    validate_name,
    validate_gender,
    validate_dob,
    correct_common_ocr_errors,
)
from backend.utils.ocr_utils import preprocess_for_ocr, preprocess_for_ocr_full

# ---------- Dependency flags ----------
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

try:
    # Keep import guarded — but we will not load models in this file (YOLO_DISABLED)
    from ultralytics import YOLO  # noqa: F401
    _ULTRALYTICS_PRESENT = True
except Exception:
    _ULTRALYTICS_PRESENT = False

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    from pyaadhaar.utils import isSecureQr
    from pyaadhaar.decode import AadhaarSecureQr
    PYAADHAAR_AVAILABLE = True
except Exception:
    PYAADHAAR_AVAILABLE = False

# ---------- Sanitizer: make JSON serializable ----------
def sanitize_value(v):
    """Convert numpy & bytes & other non-serializable types to native Python types."""
    # None, bool, int, float, str are fine
    if v is None:
        return None
    # numpy scalar
    if isinstance(v, (np.generic,)):
        try:
            return v.item()
        except Exception:
            try:
                return int(v)
            except Exception:
                return float(v)
    # numpy arrays -> list
    if isinstance(v, (np.ndarray,)):
        try:
            return v.tolist()
        except Exception:
            return str(v)
    # bytes -> base64-safe string or repr
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="ignore")
        except Exception:
            return str(v)
    # PIL.Image -> not serializable; convert to size tuple
    try:
        from PIL.Image import Image as PILImage
        if isinstance(v, PILImage):
            return {"width": v.width, "height": v.height}
    except Exception:
        pass
    # generic iterable -> sanitize each element
    if isinstance(v, (list, tuple)):
        return [sanitize_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): sanitize_value(val) for k, val in v.items()}
    # numpy bool type
    if isinstance(v, (np.bool_,)):
        return bool(v)
    # fallback to str
    if isinstance(v, (int, float, str, bool)):
        return v
    try:
        return str(v)
    except Exception:
        return None

def sanitize_json(obj):
    """Recursively sanitize an object so it is JSON serializable."""
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return sanitize_value(obj)

# ---------- EasyOCR reader (cached) ----------
_EASYREADER = None
def get_easyocr_reader(lang_list=['en'], gpu=False):
    global _EASYREADER
    if not EASYOCR_AVAILABLE:
        return None
    if _EASYREADER is None:
        try:
            _EASYREADER = easyocr.Reader(lang_list, gpu=gpu)
        except Exception:
            # CPU fallback
            _EASYREADER = easyocr.Reader(lang_list, gpu=False)
    return _EASYREADER

def easyocr_image_to_text(pil_image):
    """Return combined detected text (lowercased)."""
    if not EASYOCR_AVAILABLE:
        return ""
    try:
        reader = get_easyocr_reader()
        arr = np.array(pil_image.convert("RGB"))
        results = reader.readtext(arr, detail=1)
        texts = [t[1] if isinstance(t, (list, tuple)) and len(t) > 1 else str(t) for t in results]
        combined = " ".join(texts).strip()
        return combined.lower()
    except Exception as e:
        # Do not crash OCR failures
        return ""

def easyocr_crop_to_text(crop_pil):
    if not EASYOCR_AVAILABLE:
        return ""
    try:
        reader = get_easyocr_reader()
        arr = np.array(crop_pil.convert("RGB"))
        res = reader.readtext(arr, detail=0)
        if isinstance(res, list):
            return " ".join(res).strip()
        return str(res).strip()
    except Exception:
        return ""

# ---------- QR decode ----------
def decode_secure_qr(image_np_bgr):
    if not PYAADHAAR_AVAILABLE:
        return {"error": "QR decoding disabled - dependencies not available"}
    try:
        gray = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2GRAY)
        code = pyzbar_decode(gray)
        if not code:
            return {"error": "QR Code not found or could not be read"}
        qrData = code[0].data
        if isSecureQr(qrData):
            secure_qr = AadhaarSecureQr(int(qrData))
            decoded_data = secure_qr.decodeddata()
            return dict(decoded_data) if hasattr(decoded_data, '__dict__') else decoded_data
        else:
            return {"error": "QR code is not a valid Secure Aadhaar QR."}
    except Exception as e:
        return {"error": f"QR decoding failed: {str(e)}"}

# ---------- Aadhaar image heuristic (robust & OCR-optional) ----------
def is_aadhaar_image(image_bytes):
    """
    Heuristic to decide whether image looks like an Aadhaar card.
    Uses visual cues (aspect, size, top color band) and optional OCR boosting.
    Returns (bool_is_aadhaar, confidence_int, details_dict)
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # safe downscale
        max_dim = 1400
        w, h = image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            w, h = image.size

        aspect_ratio = (w / h) if h else 0.0
        aspect_ok = 1.1 <= aspect_ratio <= 2.9
        size_ok = min(w, h) >= 160

        # top band color heuristic
        np_img = np.array(image)
        # If image height is small, reduce sample height
        sample_h = min(80, max(10, h // 8))
        avg_top = np.mean(np_img[:sample_h, :, :], axis=(0, 1))
        # heuristic: Aadhaar has warm/orange-ish top band (R > G and R > B)
        orange_hint = bool(avg_top[0] > avg_top[1] * 0.95 and avg_top[0] > avg_top[2] * 0.9)

        keywords_found = 0
        number_found = 0
        text_snippet = ""

        if EASYOCR_AVAILABLE:
            txt = easyocr_image_to_text(image)
            text_snippet = txt[:250]
            keywords = ['aadhaar', 'aadhar', 'uidai', 'government of india', 'dob', 'date of birth', 'male', 'female']
            keywords_found = sum(1 for k in keywords if k in txt)
            number_found = len(re.findall(r"\b\d{4}\s?\d{4}\s?\d{4}\b", txt))
        else:
            txt = ""

        confidence = 0
        if aspect_ok:
            confidence += 35
        if size_ok:
            confidence += 20
        if orange_hint:
            confidence += 20
        confidence += min(keywords_found * 15, 30)
        confidence += 25 if number_found > 0 else 0

        confidence = int(min(100, confidence))
        is_aadhaar = confidence >= 40

        details = {
            "keywords_found": int(keywords_found),
            "aadhaar_numbers_found": int(number_found),
            "aspect_ratio_valid": bool(aspect_ok),
            "size_valid": bool(size_ok),
            "color_band_detected": bool(orange_hint),
            "detected_text_snippet": text_snippet
        }
        return bool(is_aadhaar), int(confidence), details

    except Exception as e:
        return False, 0, {"error": str(e)}

# ---------- OCR field extraction (crop) ----------
def ocr_text_for_label(crop_pil, label):
    if EASYOCR_AVAILABLE:
        try:
            pre = preprocess_for_ocr(crop_pil)
            return easyocr_crop_to_text(pre)
        except Exception:
            return ""
    return ""

# ---------- Main processor (single image) ----------
def process_single_image_bytes(front_bytes, back_bytes=None, do_qr_check=False, device="cpu"):
    """
    Process single Aadhaar front image bytes.
    YOLO is disabled in this build to avoid OOM crashes on small hosts.
    Returns a JSON-safe dict with keys:
      - timestamp, fraud_score, indicators, ocr_data, qr_data, assessment,
        confidence_score, extracted, aadhaar_verification
    """
    ts = datetime.datetime.now().isoformat()

    try:
        # 1) Aadhaar presence check
        is_aadhaar, aadhaar_confidence, aadhaar_details = is_aadhaar_image(front_bytes)

        if not is_aadhaar:
            payload = {
                "error": "NOT_AADHAAR",
                "message": "The uploaded image does not appear to be an Aadhaar card",
                "aadhaar_verification": aadhaar_details,
                "confidence_score": int(aadhaar_confidence),
                "timestamp": ts,
                "assessment": "INVALID_INPUT"
            }
            return sanitize_json(payload)

        # Convert to PIL & numpy
        front_image_pil = Image.open(io.BytesIO(front_bytes)).convert("RGB")
        img_np = np.array(front_image_pil)

        results = {
            "timestamp": ts,
            "fraud_score": 0,
            "indicators": [],
            "ocr_data": {},
            "qr_data": {},
            "assessment": "LOW",
            "confidence_score": int(aadhaar_confidence),
            "extracted": {},
            "aadhaar_verification": {
                "is_aadhaar_card": True,
                "confidence_score": int(aadhaar_confidence),
                "verification_details": aadhaar_details
            }
        }

        # 2) YOLO/field detection - SKIPPED (YOLO disabled)
        results["indicators"].append("⚪ INFO: Field detection (YOLO) disabled in this build.")

        # 3) Full-image OCR (if available) and heuristic extraction
        full_text = ""
        if EASYOCR_AVAILABLE:
            full_text = easyocr_image_to_text(front_image_pil)
            results["ocr_data"]["full_text"] = full_text

        # 4) Face detection - SKIPPED (models disabled)
        results["indicators"].append("⚪ INFO: Face detection disabled in this build.")

        # 5) Heuristic extraction from OCR text
        extracted_name = ""
        extracted_gender = ""
        extracted_dob = ""
        extracted_aadhaar = ""

        if full_text:
            # Aadhaar number
            m = re.search(r'\b(\d{4}\s?\d{4}\s?\d{4})\b', full_text)
            if m:
                extracted_aadhaar = m.group(1)

            # DOB
            dm = re.search(r'(\d{2}/\d{2}/\d{4})', full_text)
            if dm:
                extracted_dob = dm.group(1)

            # Gender
            if "male" in full_text:
                extracted_gender = "Male"
            elif "female" in full_text:
                extracted_gender = "Female"

            # Name heuristics (uppercase line heuristic)
            lines = [l.strip() for l in full_text.splitlines() if l.strip()]
            for ln in lines[:12]:
                # pick line with alphabets and multiple words, favorable if uppercase
                if any(c.isalpha() for c in ln) and len(ln) > 3:
                    if ln.isupper() or (len(ln.split()) <= 4 and len(ln) < 60):
                        extracted_name = extracted_name or ln

        # 6) Clean & normalise
        if extracted_aadhaar:
            extracted_aadhaar = re.sub(r'[^0-9]', '', extracted_aadhaar)

        if extracted_name:
            extracted_name = correct_common_ocr_errors(extracted_name)
        if extracted_dob:
            extracted_dob = correct_common_ocr_errors(extracted_dob)

        # 7) Validation
        a_val = validate_aadhaar_number(extracted_aadhaar) if extracted_aadhaar else "Missing"
        n_val = validate_name(extracted_name) if extracted_name else "Missing"
        g_val = validate_gender(extracted_gender) if extracted_gender else "Missing"
        d_val = validate_dob(extracted_dob) if extracted_dob else "Missing"

        if a_val == "Missing":
            results["fraud_score"] += 3
            results["indicators"].append("🔴 HIGH: Aadhaar number is missing.")
        elif "Invalid" in a_val:
            results["fraud_score"] += 3
            results["indicators"].append(f"🔴 HIGH: Aadhaar number '{extracted_aadhaar}' is {a_val}.")
        else:
            results["indicators"].append("✅ LOW: Aadhaar number extracted.")

        if n_val == "Missing":
            results["fraud_score"] += 1
            results["indicators"].append("🟡 MEDIUM: Name is missing.")
        elif "Invalid" in n_val:
            results["fraud_score"] += 1
            results["indicators"].append(f"🟡 MEDIUM: Name '{extracted_name}' is {n_val}.")
        else:
            results["indicators"].append("✅ LOW: Name format valid.")

        if d_val == "Missing":
            results["fraud_score"] += 1
            results["indicators"].append("🟡 MEDIUM: DOB is missing.")
        elif "Invalid" in d_val:
            results["fraud_score"] += 2
            results["indicators"].append(f"🔴 HIGH: DOB '{extracted_dob}' is {d_val}.")
        else:
            results["indicators"].append("✅ LOW: DOB format valid.")

        if g_val == "Missing":
            results["fraud_score"] += 1
            results["indicators"].append("🟡 MEDIUM: Gender is missing.")
        elif "Invalid" in g_val:
            results["fraud_score"] += 1
            results["indicators"].append(f"🟡 MEDIUM: Gender '{extracted_gender}' is {g_val}.")
        else:
            results["indicators"].append("✅ LOW: Gender format valid.")

        results["extracted"] = {
            "name": extracted_name or "",
            "dob": extracted_dob or "",
            "gender": extracted_gender or "",
            "aadhaar": extracted_aadhaar or ""
        }

        # 8) QR verification (attempt if requested)
        if do_qr_check:
            try:
                image_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                qr_data = decode_secure_qr(image_bgr)
                results["qr_data"] = qr_data
                if "error" not in qr_data:
                    results["indicators"].append("✅ LOW: Secure QR Code decoded successfully.")
                else:
                    results["indicators"].append(f"⚠️ QR Code: {qr_data.get('error')}")
            except Exception:
                results["indicators"].append("⚠️ QR decoding error.")

        # 9) Final assessment
        if results["fraud_score"] >= 7:
            results["assessment"] = "HIGH"
        elif results["fraud_score"] >= 3:
            results["assessment"] = "MODERATE"
        else:
            results["assessment"] = "LOW"

        # Ensure JSON-safety before returning
        return sanitize_json(results)

    except Exception as e:
        tb = traceback.format_exc()
        payload = {
            "error": "PROCESSING_FAILED",
            "message": str(e),
            "traceback": tb,
            "timestamp": datetime.datetime.now().isoformat()
        }
        return sanitize_json(payload)

# ---------- Batch processing ----------
def process_zip_bytes(zip_bytes, model_path=None, do_qr_check=False, device="cpu", max_files=None):
    """
    Processes images inside a ZIP and returns list of results (JSON-safe).
    """
    out = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            members = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
            if max_files:
                members = members[:int(max_files)]
            for name in members:
                try:
                    b = z.read(name)
                    rec = process_single_image_bytes(b, back_bytes=None, do_qr_check=do_qr_check, device=device)
                    if isinstance(rec, dict):
                        rec["filename"] = name
                    out.append(sanitize_json(rec))
                except Exception as e:
                    out.append(sanitize_json({"filename": name, "error": str(e)}))
    except Exception as e:
        out.append(sanitize_json({"error": str(e)}))
    return out
