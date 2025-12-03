# backend/utils/processor.py
import os
import io
import re
import zipfile
import datetime
import traceback
import cv2
import numpy as np
from PIL import Image, ImageDraw

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
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    from pyaadhaar.utils import isSecureQr
    from pyaadhaar.decode import AadhaarSecureQr
    PYAADHAAR_AVAILABLE = True
except Exception:
    PYAADHAAR_AVAILABLE = False

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
            # Try CPU fallback
            _EASYREADER = easyocr.Reader(lang_list, gpu=False)
    return _EASYREADER

# ---------- YOLO model caching ----------
CUSTOM_MODEL = None
FACE_MODEL = None
def load_models(device="cpu"):
    global CUSTOM_MODEL, FACE_MODEL
    if not YOLO_AVAILABLE:
        return
    try:
        if CUSTOM_MODEL is None:
            model_path = os.environ.get("MODEL_PATH", os.path.join("backend", "models", "best.pt"))
            CUSTOM_MODEL = YOLO(model_path)
            CUSTOM_MODEL.to(device)
        if FACE_MODEL is None:
            face_path = os.environ.get("FACE_MODEL_PATH", os.path.join("backend", "models", "yolov8n.pt"))
            FACE_MODEL = YOLO(face_path)
            FACE_MODEL.to(device)
    except Exception as e:
        print("⚠️ YOLO load error:", e)
        # leave models None on failure

# ---------- Helper: OCR full-image (EasyOCR) ----------
def easyocr_image_to_text(pil_image):
    """
    Returns a lowercased combined text string detected by EasyOCR for the whole image.
    """
    if not EASYOCR_AVAILABLE:
        return ""
    try:
        reader = get_easyocr_reader()
        # EasyOCR expects numpy array in RGB
        arr = np.array(pil_image.convert("RGB"))
        # reader.readtext returns list of (bbox, text, prob)
        results = reader.readtext(arr, detail=1)
        texts = [t[1] if isinstance(t, (list, tuple)) and len(t) > 1 else str(t) for t in results]
        combined = " ".join(texts)
        return combined.lower()
    except Exception as e:
        print("⚠️ easyocr_image_to_text failed:", e)
        return ""

# ---------- Helper: OCR crop (EasyOCR) ----------
def easyocr_crop_to_text(crop_pil):
    if not EASYOCR_AVAILABLE:
        return ""
    try:
        reader = get_easyocr_reader()
        arr = np.array(crop_pil.convert("RGB"))
        res = reader.readtext(arr, detail=0)  # detail=0 returns list of strings
        if isinstance(res, list):
            return " ".join(res).strip()
        return str(res).strip()
    except Exception as e:
        # fallback to empty
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

# ---------- Aadhaar image heuristic ----------
def is_aadhaar_image(image_bytes):
    """
    Heuristic to decide whether image looks like an Aadhaar card.
    Uses EasyOCR if available, otherwise uses relaxed heuristics.
    Returns (is_aadhaar_bool, confidence_score_int, details_dict)
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # safe downscale
        max_dim = 1400
        w, h = image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
            w, h = image.size

        aspect_ratio = w / h if h else 0
        aspect_ok = 1.05 <= aspect_ratio <= 2.8
        min_dim = min(w, h)
        size_ok = min_dim >= 140

        keywords = ['aadhaar', 'aadhar', 'uidai', 'unique identification authority', 'government of india', 'dob', 'date of birth', 'male', 'female']

        if EASYOCR_AVAILABLE:
            text = easyocr_image_to_text(image)
            keyword_matches = sum(1 for k in keywords if k in text)
            aadhaar_pattern = re.findall(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text)
        else:
            text = ""
            keyword_matches = 0
            aadhaar_pattern = []

        # scoring
        confidence = 0
        if keyword_matches >= 2:
            confidence += 45
        elif keyword_matches == 1:
            confidence += 20
        if aadhaar_pattern:
            confidence += 30
        if aspect_ok:
            confidence += 15
        if size_ok:
            confidence += 15

        confidence = min(100, confidence)
        return confidence >= 45, int(confidence), {
            "keywords_found": keyword_matches,
            "aadhaar_numbers_found": len(aadhaar_pattern),
            "aspect_ratio_valid": aspect_ok,
            "size_valid": size_ok,
            "detected_text_snippet": text[:250] if text else ""
        }
    except Exception as e:
        return False, 0, {"error": str(e)}

# ---------- OCR field extraction (crop) ----------
def ocr_text_for_label(crop_pil, label):
    """
    Returns text string for a crop, using EasyOCR if available, otherwise empty.
    Label parameter kept for compatibility with whitelist config.
    """
    if EASYOCR_AVAILABLE:
        return easyocr_crop_to_text(preprocess_for_ocr(crop_pil))
    else:
        return ""

# ---------- Main processor (single image) ----------
def process_single_image_bytes(front_bytes, back_bytes=None, do_qr_check=False, device="cpu"):
    """
    Unified processing function: uses EasyOCR, YOLO (if available), QR (if available).
    Returns JSON-serializable dict.
    """
    ts = datetime.datetime.now().isoformat()

    # 1) Aadhaar card check
    is_aadhaar, aadhaar_confidence, aadhaar_details = is_aadhaar_image(front_bytes)

    if not is_aadhaar:
        return {
            "error": "NOT_AADHAAR",
            "message": "The uploaded image does not appear to be an Aadhaar card",
            "aadhaar_verification": aadhaar_details,
            "confidence_score": aadhaar_confidence,
            "timestamp": ts,
            "assessment": "INVALID_INPUT"
        }

    # 2) Load models if available
    try:
        load_models(device)
    except Exception as e:
        # continue even if model loading fails
        print("⚠️ load_models issue:", e)

    # Convert bytes to PIL and numpy
    front_image_pil = Image.open(io.BytesIO(front_bytes)).convert("RGB")
    img_np = np.array(front_image_pil)

    results = {
        "timestamp": ts,
        "fraud_score": 0,
        "indicators": [],
        "ocr_data": {},
        "qr_data": {},
        "assessment": "LOW",
        "confidence_score": aadhaar_confidence,
        "extracted": {},
        "aadhaar_verification": {
            "is_aadhaar_card": True,
            "confidence_score": aadhaar_confidence,
            "verification_details": aadhaar_details
        }
    }

    # 3) YOLO field detection (if available)
    if YOLO_AVAILABLE and CUSTOM_MODEL is not None:
        try:
            dets = CUSTOM_MODEL(img_np, conf=0.25, verbose=False)[0]
            if dets.boxes:
                for box in dets.boxes:
                    class_id = int(box.cls[0])
                    label = CUSTOM_MODEL.names[class_id]
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    crop = front_image_pil.crop((x1, y1, x2, y2))
                    text = ocr_text_for_label(crop, label)
                    if text:
                        results["ocr_data"][label] = text
        except Exception as e:
            results["indicators"].append(f"Field detection error: {str(e)}")
            results["fraud_score"] += 3
    else:
        # No YOLO: do a full-image OCR and try to heuristically extract fields
        if EASYOCR_AVAILABLE:
            full_text = easyocr_image_to_text(front_image_pil)
            results["ocr_data"]["full_text"] = full_text

    # 4) Face detection (if face model available)
    if YOLO_AVAILABLE and FACE_MODEL is not None:
        try:
            face_res = FACE_MODEL(img_np, classes=[0], conf=0.4, verbose=False)[0]
            if len(face_res.boxes) > 0:
                results["indicators"].append("✅ LOW: Face detected on card.")
            else:
                results["fraud_score"] += 3
                results["indicators"].append("🔴 HIGH: No face detected on the card.")
        except Exception as e:
            results["indicators"].append("⚠️ Face detection failed.")
    else:
        results["indicators"].append("⚪ INFO: Face model not available.")

    # 5) Extract fields from ocr_data heuristically
    ocr_storage = results["ocr_data"]
    extracted_name = ""
    extracted_gender = ""
    extracted_dob = ""
    extracted_aadhaar = ""

    # If YOLO produced labeled fields, use them
    if ocr_storage:
        for k, v in ocr_storage.items():
            lk = k.lower()
            txt = v.strip()
            if not extracted_aadhaar and any(x in lk for x in ["aadhaar", "aadhar", "uid", "number", "id"]):
                extracted_aadhaar = txt
            if not extracted_name and "name" in lk:
                extracted_name = txt
            if not extracted_gender and "gender" in lk:
                extracted_gender = txt
            if not extracted_dob and ("dob" in lk or "date" in lk or "birth" in lk):
                extracted_dob = txt

    # If only full_text exists, try regex extraction
    if not extracted_aadhaar and "full_text" in ocr_storage:
        txt = ocr_storage["full_text"]
        m = re.search(r'\b(\d{4}\s?\d{4}\s?\d{4})\b', txt)
        if m:
            extracted_aadhaar = m.group(1)

        # Name heuristics: find lines with uppercase words and length
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        for ln in lines[:12]:
            if len(ln) > 3 and ln.isupper() and any(c.isalpha() for c in ln):
                extracted_name = extracted_name or ln

        # DOB
        dm = re.search(r'(\d{2}/\d{2}/\d{4})', txt)
        if dm:
            extracted_dob = dm.group(1)

        # Gender
        if "male" in txt.lower():
            extracted_gender = "Male"
        elif "female" in txt.lower():
            extracted_gender = "Female"

    # 6) Clean & correct fields
    if extracted_aadhaar:
        extracted_aadhaar = re.sub(r'[^0-9]', '', extracted_aadhaar)
        extracted_aadhaar = re.sub(r'\s+', '', extracted_aadhaar)

    if extracted_name:
        extracted_name = correct_common_ocr_errors(extracted_name)

    if extracted_dob:
        extracted_dob = correct_common_ocr_errors(extracted_dob)

    # 7) Validation
    if extracted_aadhaar:
        a_val = validate_aadhaar_number(extracted_aadhaar)
    else:
        a_val = "Missing"
    n_val = validate_name(extracted_name) if extracted_name else "Missing"
    g_val = validate_gender(extracted_gender) if extracted_gender else "Missing"
    d_val = validate_dob(extracted_dob) if extracted_dob else "Missing"

    # scoring and indicators
    if a_val == "Missing":
        results["fraud_score"] += 3
        results["indicators"].append("🔴 HIGH: Aadhaar number is missing.")
    elif "Invalid" in a_val:
        results["fraud_score"] += 3
        results["indicators"].append(f"🔴 HIGH: Aadhaar number '{extracted_aadhaar}' is {a_val}.")
    else:
        results["indicators"].append(f"✅ LOW: Aadhaar number extracted.")

    if n_val == "Missing":
        results["fraud_score"] += 1
        results["indicators"].append("🟡 MEDIUM: Name is missing.")
    elif "Invalid" in n_val:
        results["fraud_score"] += 1
        results["indicators"].append(f"🟡 MEDIUM: Name '{extracted_name}' is {n_val}.")
    else:
        results["indicators"].append(f"✅ LOW: Name format valid.")

    if d_val == "Missing":
        results["fraud_score"] += 1
        results["indicators"].append("🟡 MEDIUM: DOB is missing.")
    elif "Invalid" in d_val:
        results["fraud_score"] += 2
        results["indicators"].append(f"🔴 HIGH: DOB '{extracted_dob}' is {d_val}.")
    else:
        results["indicators"].append(f"✅ LOW: DOB format valid.")

    if g_val == "Missing":
        results["fraud_score"] += 1
        results["indicators"].append("🟡 MEDIUM: Gender is missing.")
    elif "Invalid" in g_val:
        results["fraud_score"] += 1
        results["indicators"].append(f"🟡 MEDIUM: Gender '{extracted_gender}' is {g_val}.")
    else:
        results["indicators"].append(f"✅ LOW: Gender format valid.")

    # fill extracted
    results["extracted"] = {
        "name": extracted_name,
        "dob": extracted_dob,
        "gender": extracted_gender,
        "aadhaar": extracted_aadhaar
    }

    # 8) QR verification if requested
    if do_qr_check:
        try:
            image_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            qr_data = decode_secure_qr(image_bgr)
            results["qr_data"] = qr_data
            if "error" not in qr_data:
                results["indicators"].append("✅ LOW: Secure QR Code decoded successfully.")
            else:
                results["indicators"].append(f"⚠️ QR Code: {qr_data.get('error')}")
        except Exception as e:
            results["indicators"].append("⚠️ QR decoding error.")

    # 9) Final assessment
    if results["fraud_score"] >= 7:
        results["assessment"] = "HIGH"
    elif results["fraud_score"] >= 3:
        results["assessment"] = "MODERATE"
    else:
        results["assessment"] = "LOW"

    return results

# ---------- Batch processing ----------
def process_zip_bytes(zip_bytes, model_path=None, do_qr_check=False, device="cpu", max_files=None):
    """
    Processes images inside a ZIP and returns list of results.
    """
    out = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            members = [n for n in z.namelist() if n.lower().endswith((".jpg",".jpeg",".png"))]
            if max_files:
                members = members[:int(max_files)]
            for name in members:
                try:
                    b = z.read(name)
                    rec = process_single_image_bytes(b, back_bytes=None, do_qr_check=do_qr_check, device=device)
                    rec["filename"] = name
                    out.append(rec)
                except Exception as e:
                    out.append({"filename": name, "error": str(e)})
    except Exception as e:
        out.append({"error": str(e)})
    return out
