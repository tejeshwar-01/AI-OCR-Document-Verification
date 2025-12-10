# backend/utils/processor.py
"""
Processor utilities - flexible build:
- YOLO is OFF by default (safe for small hosts). Enable via env ENABLE_YOLO=1 and set YOLO_MODEL_PATH.
- Uses EasyOCR if present, falls back to pytesseract if present, or returns best-effort heuristics.
- All outputs sanitized for JSON.
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

# ---------- Project utilities (assumed present) ----------
from backend.utils.verification_rules import (
    validate_aadhaar_number,
    validate_name,
    validate_gender,
    validate_dob,
    correct_common_ocr_errors,
)
from backend.utils.ocr_utils import preprocess_for_ocr, preprocess_for_ocr_full

# ---------- Feature flags & dependency probes ----------
# YOLO: disabled by default; enable via env ENABLE_YOLO=1 and YOLO_MODEL_PATH=/path/to/model.pt
ENABLE_YOLO_ENV = os.getenv("ENABLE_YOLO", "0")
YOLO_ENABLED_REQUESTED = ENABLE_YOLO_ENV in ("1", "true", "True", "yes", "YES")

try:
    from ultralytics import YOLO  # type: ignore
    _ULTRALYTICS_PRESENT = True
except Exception:
    _ULTRALYTICS_PRESENT = False

# EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

# pytesseract fallback
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

# pyzbar + pyaadhaar for QR
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    from pyaadhaar.utils import isSecureQr
    from pyaadhaar.decode import AadhaarSecureQr
    PYAADHAAR_AVAILABLE = True
except Exception:
    PYAADHAAR_AVAILABLE = False

# ---------- Sanitizer: make JSON serializable ----------
def sanitize_value(v):
    if v is None:
        return None
    if isinstance(v, (np.generic,)):
        try:
            return v.item()
        except Exception:
            try:
                return int(v)
            except Exception:
                return float(v)
    if isinstance(v, (np.ndarray,)):
        try:
            return v.tolist()
        except Exception:
            return str(v)
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="ignore")
        except Exception:
            return str(v)
    try:
        from PIL.Image import Image as PILImage
        if isinstance(v, PILImage):
            return {"width": v.width, "height": v.height}
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return [sanitize_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): sanitize_value(val) for k, val in v.items()}
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (int, float, str, bool)):
        return v
    try:
        return str(v)
    except Exception:
        return None

def sanitize_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return sanitize_value(obj)

# ---------- EasyOCR reader cache ----------
_EASYREADER = None
def get_easyocr_reader(lang_list=['en'], gpu=False):
    global _EASYREADER
    if not EASYOCR_AVAILABLE:
        return None
    if _EASYREADER is None:
        try:
            _EASYREADER = easyocr.Reader(lang_list, gpu=gpu)
        except Exception:
            _EASYREADER = easyocr.Reader(lang_list, gpu=False)
    return _EASYREADER

def easyocr_image_to_text(pil_image):
    if not EASYOCR_AVAILABLE:
        return ""
    try:
        reader = get_easyocr_reader()
        arr = np.array(pil_image.convert("RGB"))
        results = reader.readtext(arr, detail=1)
        texts = [t[1] if isinstance(t, (list, tuple)) and len(t) > 1 else str(t) for t in results]
        combined = " ".join(texts).strip()
        return combined.lower()
    except Exception:
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

# ---------- pytesseract helpers ----------
def pytesseract_image_to_text(pil_image):
    if not PYTESSERACT_AVAILABLE:
        return ""
    try:
        arr = np.array(preprocess_for_ocr_full(pil_image) if 'preprocess_for_ocr_full' in globals() else pil_image)
        # pytesseract expects RGB or grayscale PIL image; pass PIL directly if available
        pil = pil_image if isinstance(pil_image, Image.Image) else Image.fromarray(arr)
        text = pytesseract.image_to_string(pil)
        return text.lower().strip()
    except Exception:
        return ""

# ---------- YOLO model lazy loader ----------
_YOLO_MODEL = None
def get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    if not YOLO_ENABLED_REQUESTED or not _ULTRALYTICS_PRESENT:
        return None
    model_path = os.getenv("YOLO_MODEL_PATH", "").strip()
    if not model_path or not os.path.exists(model_path):
        # no model found
        return None
    try:
        _YOLO_MODEL = YOLO(model_path)
        return _YOLO_MODEL
    except Exception:
        _YOLO_MODEL = None
        return None

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
            # decoded_data might be dict-like or object; normalize
            if hasattr(decoded_data, "__dict__"):
                return dict(decoded_data.__dict__)
            elif isinstance(decoded_data, dict):
                return decoded_data
            else:
                return {"data": str(decoded_data)}
        else:
            return {"error": "QR code is not a valid Secure Aadhaar QR."}
    except Exception as e:
        return {"error": f"QR decoding failed: {str(e)}"}

# ---------- Aadhaar image heuristic ----------
def is_aadhaar_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        max_dim = 1400
        w, h = image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            w, h = image.size

        aspect_ratio = (w / h) if h else 0.0
        aspect_ok = 1.1 <= aspect_ratio <= 2.9
        size_ok = min(w, h) >= 140

        np_img = np.array(image)
        sample_h = min(80, max(10, h // 8))
        avg_top = np.mean(np_img[:sample_h, :, :], axis=(0, 1))
        orange_hint = bool(avg_top[0] > avg_top[1] * 0.95 and avg_top[0] > avg_top[2] * 0.9)

        keywords_found = 0
        number_found = 0
        text_snippet = ""

        # Use any available OCR to get text hints
        txt = ""
        if EASYOCR_AVAILABLE:
            txt = easyocr_image_to_text(image)
        elif PYTESSERACT_AVAILABLE:
            txt = pytesseract_image_to_text(image)
        txt = txt or ""
        text_snippet = txt[:300]
        keywords = ['aadhaar', 'aadhar', 'uidai', 'government of india', 'dob', 'date of birth', 'male', 'female']
        keywords_found = sum(1 for k in keywords if k in txt)
        number_found = len(re.findall(r"\b\d{4}\s?\d{4}\s?\d{4}\b", txt))

        confidence = 0
        confidence += 35 if aspect_ok else 0
        confidence += 20 if size_ok else 0
        confidence += 20 if orange_hint else 0
        confidence += min(keywords_found * 12, 30)
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

# ---------- OCR helpers for crops ----------
def ocr_text_for_label(crop_pil, label=""):
    # Attempt EasyOCR first, then pytesseract. Preprocess if helper available.
    if EASYOCR_AVAILABLE:
        try:
            pre = preprocess_for_ocr(crop_pil)
            txt = easyocr_crop_to_text(pre) or ""
            return txt.strip()
        except Exception:
            pass
    if PYTESSERACT_AVAILABLE:
        try:
            pre = preprocess_for_ocr(crop_pil)
            txt = pytesseract_image_to_text(pre) or ""
            return txt.strip()
        except Exception:
            pass
    # last resort: plain convert
    try:
        return str(crop_pil.convert("L").tobytes()[:0])  # empty fallback
    except Exception:
        return ""

# ---------- Strong regex helpers ----------
AADHAAR_RE = re.compile(r'(\d{4}\s?\d{4}\s?\d{4})')
DOB_RE = re.compile(r'(\d{2}[\/\-\.\s]\d{2}[\/\-\.\s]\d{4})')
YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')

def find_aadhaar_in_text(txt):
    if not txt:
        return ""
    m = AADHAAR_RE.search(txt)
    return m.group(1) if m else ""

def find_dob_in_text(txt):
    if not txt:
        return ""
    m = DOB_RE.search(txt)
    if m:
        return m.group(1)
    # Try more relaxed patterns (dd mm yyyy)
    m2 = re.search(r'(\d{2}\s?\d{2}\s?\d{4})', txt)
    if m2:
        s = m2.group(1)
        # insert slashes
        return f"{s[0:2]}/{s[2:4]}/{s[4:8]}"
    return ""

def guess_name_from_text(txt):
    if not txt:
        return ""
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    # prefer lines that are uppercase or contain 2-4 words and letters-only (with minor chars)
    candidates = []
    for ln in lines[:20]:
        if any(c.isalpha() for c in ln) and len(ln) > 3 and len(ln) < 80:
            # avoid lines with 'dob', 'male', 'female', 'government'
            low = ln.lower()
            if any(skip in low for skip in ['dob', 'date of birth', 'male', 'female', 'government', 'aadhaar', 'uidai']):
                continue
            candidates.append(ln)
    # prefer uppercase lines first
    for c in candidates:
        if c.isupper() and len(c.split()) <= 5:
            return c
    # fallback to first reasonable candidate
    return candidates[0] if candidates else ""

# ---------- Main processor (single image) ----------
def process_single_image_bytes(front_bytes, back_bytes=None, do_qr_check=False, device="cpu"):
    ts = datetime.datetime.now().isoformat()
    try:
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

        front_image_pil = Image.open(io.BytesIO(front_bytes)).convert("RGB")
        img_np_rgb = np.array(front_image_pil)
        img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

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

        # YOLO field detection (if enabled & model present)
        yolo_model = get_yolo_model()
        fields = {}  # label -> PIL crop
        if yolo_model is None:
            results["indicators"].append("⚪ INFO: Field detection (YOLO) disabled in this build.")
        else:
            try:
                # run detection (use small size inference to reduce memory)
                preds = yolo_model(img_np_rgb, device=device)
                # take first result
                boxes = getattr(preds[0], "boxes", None)
                names = getattr(preds[0], "names", None) or {}
                if boxes is not None:
                    for b in boxes:
                        try:
                            cls = int(b.cls.cpu().numpy()) if hasattr(b, "cls") else int(b.data[5])
                        except Exception:
                            cls = int(getattr(b, "cls", 0))
                        label = str(names.get(cls, cls)) if names is not None else str(cls)
                        xy = getattr(b, "xyxy", None)
                        if xy is None:
                            coords = b.xyxy[0] if hasattr(b, "xyxy") else None
                        else:
                            coords = xy[0]
                        if coords is not None:
                            x1, y1, x2, y2 = map(int, coords[:4])
                            # clamp
                            h, w = img_np_rgb.shape[:2]
                            x1, x2 = max(0, x1), min(w, x2)
                            y1, y2 = max(0, y1), min(h, y2)
                            crop = front_image_pil.crop((x1, y1, x2, y2))
                            # Normalize common label names
                            norm_label = label.lower()
                            if "name" in norm_label:
                                fields["name"] = crop
                            elif "dob" in norm_label or "date" in norm_label:
                                fields["dob"] = crop
                            elif "gender" in norm_label:
                                fields["gender"] = crop
                            elif "aadhaar" in norm_label or "uid" in norm_label or "number" in norm_label:
                                fields["aadhaar"] = crop
                            else:
                                # keep generic
                                fields[label] = crop
                if fields:
                    results["indicators"].append("✅ LOW: YOLO field detection ran and returned crops.")
                else:
                    results["indicators"].append("⚪ INFO: YOLO ran but no labelled fields detected.")
            except Exception as e:
                results["indicators"].append(f"⚠️ YOLO detection error: {str(e)}")

        # Full-image OCR (for heuristics & fallback)
        full_text = ""
        if EASYOCR_AVAILABLE:
            try:
                full_text = easyocr_image_to_text(front_image_pil)
                results["ocr_data"]["full_text"] = full_text
            except Exception:
                full_text = ""
        elif PYTESSERACT_AVAILABLE:
            try:
                full_text = pytesseract_image_to_string(front_image_pil) if False else pytesseract_image_to_text(front_image_pil)
                results["ocr_data"]["full_text"] = full_text
            except Exception:
                full_text = ""
        else:
            results["ocr_data"]["full_text"] = ""

        # Face detection info (kept disabled by default)
        results["indicators"].append("⚪ INFO: Face detection disabled in this build.")

        # ---------- Extraction logic ----------
        extracted_name = ""
        extracted_gender = ""
        extracted_dob = ""
        extracted_aadhaar = ""

        # 1) If YOLO gave crops, OCR those specific crops.
        if fields.get("aadhaar"):
            extracted_aadhaar = ocr_text_for_label(fields["aadhaar"], "aadhaar")
        if fields.get("dob"):
            extracted_dob = ocr_text_for_label(fields["dob"], "dob")
        if fields.get("gender"):
            extracted_gender = ocr_text_for_label(fields["gender"], "gender")
        if fields.get("name"):
            extracted_name = ocr_text_for_label(fields["name"], "name")

        # 2) If any field missing, fallback to searching full_text
        if not extracted_aadhaar:
            found = find_aadhaar_in_text(full_text)
            extracted_aadhaar = found or ""
        if not extracted_dob:
            found = find_dob_in_text(full_text)
            extracted_dob = found or ""
        if not extracted_gender:
            if "male" in full_text:
                extracted_gender = "Male"
            elif "female" in full_text:
                extracted_gender = "Female"
        if not extracted_name:
            guess = guess_name_from_text(results["ocr_data"].get("full_text", "") or full_text)
            extracted_name = guess or ""

        # 3) Clean & normalize
        if extracted_aadhaar:
            extracted_aadhaar = re.sub(r'[^0-9]', '', extracted_aadhaar)
        if extracted_name:
            extracted_name = correct_common_ocr_errors(extracted_name).strip()
        if extracted_dob:
            extracted_dob = correct_common_ocr_errors(extracted_dob).strip()

        # 4) Validation & scoring (weights tuned to match previous behavior)
        a_val = validate_aadhaar_number(extracted_aadhaar) if extracted_aadhaar else "Missing"
        n_val = validate_name(extracted_name) if extracted_name else "Missing"
        g_val = validate_gender(extracted_gender) if extracted_gender else "Missing"
        d_val = validate_dob(extracted_dob) if extracted_dob else "Missing"

        # Aadhaar scoring (higher weight)
        if a_val == "Missing":
            results["fraud_score"] += 3
            results["indicators"].append("🔴 HIGH: Aadhaar number is missing.")
        elif "Invalid" in a_val:
            results["fraud_score"] += 3
            results["indicators"].append(f"🔴 HIGH: Aadhaar number '{extracted_aadhaar}' is {a_val}.")
        else:
            results["indicators"].append("✅ LOW: Aadhaar number extracted.")

        # Name scoring
        if n_val == "Missing":
            results["fraud_score"] += 1
            results["indicators"].append("🟡 MEDIUM: Name is missing.")
        elif "Invalid" in n_val:
            results["fraud_score"] += 1
            results["indicators"].append(f"🟡 MEDIUM: Name '{extracted_name}' is {n_val}.")
        else:
            results["indicators"].append("✅ LOW: Name format valid.")

        # DOB scoring
        if d_val == "Missing":
            results["fraud_score"] += 1
            results["indicators"].append("🟡 MEDIUM: DOB is missing.")
        elif "Invalid" in d_val:
            results["fraud_score"] += 2
            results["indicators"].append(f"🔴 HIGH: DOB '{extracted_dob}' is {d_val}.")
        else:
            results["indicators"].append("✅ LOW: DOB format valid.")

        # Gender scoring
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

        # 5) QR verification (optional)
        if do_qr_check:
            try:
                qr_data = decode_secure_qr(img_np_bgr)
                results["qr_data"] = qr_data
                if "error" not in qr_data:
                    results["indicators"].append("✅ LOW: Secure QR Code decoded successfully.")
                else:
                    results["indicators"].append(f"⚠️ QR Code: {qr_data.get('error')}")
            except Exception:
                results["indicators"].append("⚠️ QR decoding error.")

        # 6) Final assessment
        if results["fraud_score"] >= 7:
            results["assessment"] = "HIGH"
        elif results["fraud_score"] >= 3:
            results["assessment"] = "MODERATE"
        else:
            results["assessment"] = "LOW"

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
