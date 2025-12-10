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
ENABLE_YOLO_ENV = os.getenv("ENABLE_YOLO", "0")
YOLO_ENABLED_REQUESTED = ENABLE_YOLO_ENV in ("1", "true", "True", "yes", "YES")

try:
    from ultralytics import YOLO
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

# QR dependencies
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    from pyaadhaar.utils import isSecureQr
    from pyaadhaar.decode import AadhaarSecureQr
    PYAADHAAR_AVAILABLE = True
except Exception:
    PYAADHAAR_AVAILABLE = False


# -------------------- FIX A: Improved NAME OCR --------------------
def clean_name_text(txt):
    """Remove unwanted characters and normalize spacing."""
    txt = re.sub(r"[^A-Za-z\s.]", "", txt)
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt.strip()


def ocr_name_strict_only(pil_img):
    """
    Very accurate single-line OCR for Aadhaar NAME field.
    No extra models needed.
    """
    if not PYTESSERACT_AVAILABLE:
        return ""

    import cv2, numpy as np
    img = np.array(pil_img.convert("RGB"))[:, :, ::-1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

    proc = Image.fromarray(gray)

    config = "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz. "

    try:
        txt = pytesseract.image_to_string(proc, config=config)
    except:
        txt = pytesseract.image_to_string(proc)

    return clean_name_text(txt)
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
        return {str(k): sanitize_json(v) for k, v in v.items()}
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


# ---------- EasyOCR wrapper ----------
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
        texts = [
            t[1] if isinstance(t, (list, tuple)) and len(t) > 1 else str(t)
            for t in results
        ]
        return " ".join(texts).strip().lower()
    except Exception:
        return ""


def easyocr_crop_to_text(crop_pil):
    if not EASYOCR_AVAILABLE:
        return ""
    try:
        reader = get_easyocr_reader()
        arr = np.array(crop_pil.convert("RGB"))
        out = reader.readtext(arr, detail=0)
        if isinstance(out, list):
            return " ".join(out).strip()
        return str(out).strip()
    except Exception:
        return ""


# ---------- DOB, Gender OCR (original strict versions) ----------
def ocr_name_strict(pil_img):
    """Old strict OCR (kept for compatibility)."""
    return ocr_name_strict_only(pil_img)


def preprocess_dob_crop(pil_img, target_width=800):
    try:
        img = np.array(pil_img.convert("RGB"))[:, :, ::-1]
        h, w = img.shape[:2]
        if w < target_width:
            sc = target_width / float(w)
            img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    except Exception:
        try:
            return np.array(pil_img.convert("L"))
        except:
            return None


def ocr_dob_strict(pil_img):
    if not PYTESSERACT_AVAILABLE:
        return ""
    try:
        proc = preprocess_dob_crop(pil_img)
        if proc is None:
            return ""
        proc_pil = Image.fromarray(proc)

        config = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/-."
        txt = pytesseract.image_to_string(proc_pil, config=config).strip()

        txt = txt.replace(".", "/").replace("-", "/")
        txt = re.sub(r"[^\d/]", "", txt)

        # Format dd/mm/yyyy
        m = re.search(r'(\d{2})/?(\d{2})/?(\d{4})', txt)
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else txt
    except:
        return ""


def preprocess_gender_crop(pil_img, target_width=600):
    try:
        img = np.array(pil_img.convert("RGB"))[:, :, ::-1]
        h, w = img.shape[:2]
        if w < target_width:
            sc = target_width / float(w)
            img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    except:
        try:
            return np.array(pil_img.convert("L"))
        except:
            return None


def ocr_gender_strict(pil_img):
    if EASYOCR_AVAILABLE:
        try:
            txt = easyocr_crop_to_text(pil_img).lower()
            if "male" in txt:
                return "Male"
            if "female" in txt:
                return "Female"
        except:
            pass

    if not PYTESSERACT_AVAILABLE:
        return ""

    try:
        proc = preprocess_gender_crop(pil_img)
        proc_pil = Image.fromarray(proc)

        config = "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        txt = pytesseract.image_to_string(proc_pil, config=config).lower()

        if "male" in txt:
            return "Male"
        if "female" in txt:
            return "Female"

        # fallback m/f
        letters = re.sub(r"[^a-z]", "", txt)
        if letters == "m":
            return "Male"
        if letters == "f":
            return "Female"
        return ""
    except:
        return ""


# ---------- Heuristic crops ----------
def heuristic_name_crop(full_pil):
    w, h = full_pil.size
    return full_pil.crop((int(w*0.05), int(h*0.18), int(w*0.75), int(h*0.33)))


def heuristic_dob_crop(full_pil):
    w, h = full_pil.size
    return full_pil.crop((int(w*0.05), int(h*0.30), int(w*0.75), int(h*0.42)))


def heuristic_gender_crop(full_pil):
    w, h = full_pil.size
    return full_pil.crop((int(w*0.60), int(h*0.30), int(w*0.90), int(h*0.44)))


# ---------- pytesseract helper ----------
def pytesseract_image_to_text(pil_image):
    if not PYTESSERACT_AVAILABLE:
        return ""
    try:
        arr = np.array(preprocess_for_ocr_full(pil_image))
        pil = Image.fromarray(arr)
        return pytesseract.image_to_string(pil).lower().strip()
    except:
        return ""


# ---------- YOLO loader ----------
_YOLO_MODEL = None
def get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL:
        return _YOLO_MODEL
    if not YOLO_ENABLED_REQUESTED or not _ULTRALYTICS_PRESENT:
        return None
    path = os.getenv("YOLO_MODEL_PATH", "").strip()
    if not path or not os.path.exists(path):
        return None
    try:
        _YOLO_MODEL = YOLO(path)
        return _YOLO_MODEL
    except:
        _YOLO_MODEL = None
        return None


# ---------- QR Decode ----------
def decode_secure_qr(image_np_bgr):
    if not PYAADHAAR_AVAILABLE:
        return {"error": "QR decoding disabled"}
    try:
        gray = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2GRAY)
        codes = pyzbar_decode(gray)
        if not codes:
            return {"error": "QR Code not found"}

        raw = codes[0].data
        if isSecureQr(raw):
            obj = AadhaarSecureQr(int(raw)).decodeddata()
            if hasattr(obj, "__dict__"):
                return dict(obj.__dict__)
            if isinstance(obj, dict):
                return obj
            return {"data": str(obj)}
        return {"error": "Invalid Secure QR"}
    except Exception as e:
        return {"error": str(e)}


# ---------- Aadhaar heuristic detection ----------
def is_aadhaar_image(image_bytes):
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(pil)
        w, h = pil.size

        # aspect ratio filter
        aspect = w / h
        size_ok = min(w, h) >= 150
        aspect_ok = 1.1 <= aspect <= 2.9

        # top-color orange band
        sample = min(80, h//8)
        avg = np.mean(np_img[:sample], axis=(0,1))
        orange = avg[0] > avg[1]*0.95 and avg[0] > avg[2]*0.9

        # OCR quick hints
        txt = ""
        if EASYOCR_AVAILABLE:
            txt = easyocr_image_to_text(pil)
        elif PYTESSERACT_AVAILABLE:
            txt = pytesseract_image_to_text(pil)
        txt = txt or ""

        keywords = ["aadhaar","aadhar","uidai","dob","male","female"]
        words = sum(k in txt for k in keywords)
        nums = len(re.findall(r"\d{4}\s?\d{4}\s?\d{4}", txt))

        score = 0
        score += 35 if aspect_ok else 0
        score += 20 if size_ok else 0
        score += 20 if orange else 0
        score += min(words*10, 30)
        score += 20 if nums else 0

        return score >= 40, min(score,100), {
            "aspect_ok": bool(aspect_ok),
            "size_ok": bool(size_ok),
            "orange_band": bool(orange),
            "keywords_found": words,
            "numbers_found": nums,
            "snippet": txt[:250]
        }

    except Exception as e:
        return False, 0, {"error": str(e)}


# ---------- Regex helpers ----------
AADHAAR_RE = re.compile(r"(\d{4}\s?\d{4}\s?\d{4})")
DOB_RE = re.compile(r"(\d{2}[/-]\d{2}[/-]\d{4})")

def find_aadhaar_in_text(txt):
    m = AADHAAR_RE.search(txt or "")
    return m.group(1) if m else ""

def find_dob_in_text(txt):
    txt = txt or ""
    m = DOB_RE.search(txt)
    if m: return m.group(1)

    m2 = re.search(r"(\d{2})\s?(\d{2})\s?(\d{4})", txt)
    if m2:
        return f"{m2.group(1)}/{m2.group(2)}/{m2.group(3)}"
    return ""

def guess_name_from_text(txt):
    if not txt:
        return ""
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    cands = []
    for ln in lines[:20]:
        if (
            any(c.isalpha() for c in ln)
            and 3 < len(ln) < 80
            and not any(s in ln.lower() for s in ["dob","male","female","government","uidai"])
        ):
            cands.append(ln)

    for x in cands:
        if x.isupper() and len(x.split()) <= 5:
            return x
    return cands[0] if cands else ""
# ---------- Main processor (single image) ----------
def process_single_image_bytes(front_bytes, back_bytes=None, do_qr_check=False, device="cpu"):
    ts = datetime.datetime.now().isoformat()

    try:
        # Aadhaar heuristic
        is_aadhaar, confidence, info = is_aadhaar_image(front_bytes)

        if not is_aadhaar:
            return sanitize_json({
                "error": "NOT_AADHAAR",
                "message": "The uploaded image does not appear to be an Aadhaar card",
                "confidence_score": confidence,
                "aadhaar_verification": info,
                "timestamp": ts,
                "assessment": "INVALID_INPUT"
            })

        # Load image
        front_pil = Image.open(io.BytesIO(front_bytes)).convert("RGB")
        img_np_rgb = np.array(front_pil)
        img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

        results = {
            "timestamp": ts,
            "fraud_score": 0,
            "assessment": "LOW",
            "confidence_score": confidence,
            "ocr_data": {},
            "indicators": [],
            "extracted": {},
            "qr_data": {},
            "aadhaar_verification": {
                "is_aadhaar_card": True,
                "confidence_score": confidence,
                "verification_details": info
            }
        }

        # ---------- YOLO detection ----------
        fields = {}
        yolo_model = get_yolo_model()

        if not yolo_model:
            results["indicators"].append("⚪ INFO: Field detection (YOLO) disabled in this build.")
        else:
            try:
                preds = yolo_model(img_np_rgb, device=device)
                boxes = preds[0].boxes
                names = preds[0].names

                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls.cpu().numpy())
                        label = names.get(cls, str(cls))
                        xyxy = box.xyxy[0]
                        x1, y1, x2, y2 = map(int, xyxy)

                        h, w = img_np_rgb.shape[:2]
                        x1, x2 = max(0, x1), min(w, x2)
                        y1, y2 = max(0, y1), min(h, y2)

                        crop = front_pil.crop((x1, y1, x2, y2))
                        lbl = label.lower()

                        if "name" in lbl:
                            fields["name"] = crop
                        elif "dob" in lbl:
                            fields["dob"] = crop
                        elif "gender" in lbl:
                            fields["gender"] = crop
                        elif "aadhaar" in lbl or "uid" in lbl:
                            fields["aadhaar"] = crop

                if fields:
                    results["indicators"].append("✅ LOW: YOLO field detection ran.")
                else:
                    results["indicators"].append("⚪ INFO: YOLO detected no labelled crops.")

            except Exception as e:
                results["indicators"].append(f"⚠️ YOLO detection error: {str(e)}")

        # ---------- FULL OCR TEXT ----------
        full_text = ""
        if EASYOCR_AVAILABLE:
            full_text = easyocr_image_to_text(front_pil)
        elif PYTESSERACT_AVAILABLE:
            full_text = pytesseract_image_to_text(front_pil)

        results["ocr_data"]["full_text"] = full_text

        extracted_name = ""
        extracted_dob = ""
        extracted_gender = ""
        extracted_aadhaar = ""

        # ---------- Aadhaar number ----------
        if fields.get("aadhaar"):
            extracted_aadhaar = ocr_text_for_label(fields["aadhaar"])
        if not extracted_aadhaar:
            extracted_aadhaar = find_aadhaar_in_text(full_text)
        extracted_aadhaar = re.sub(r"[^0-9]", "", extracted_aadhaar)

        # ---------- DOB ----------
        if fields.get("dob"):
            extracted_dob = ocr_dob_strict(fields["dob"])
        if not extracted_dob:
            extracted_dob = find_dob_in_text(full_text)
        extracted_dob = extracted_dob.strip()

        # ---------- Gender ----------
        if fields.get("gender"):
            extracted_gender = ocr_gender_strict(fields["gender"])
        if not extracted_gender:
            if "male" in full_text:
                extracted_gender = "Male"
            elif "female" in full_text:
                extracted_gender = "Female"

        # ---------- NAME (fixed) ----------
        if fields.get("name"):
            extracted_name = ocr_name_strict_only(fields["name"])

        # Fallback: Infer from full OCR text
        if not extracted_name:
            extracted_name = guess_name_from_text(full_text)

        # Final fallback: heuristic crop
        if not extracted_name:
            extracted_name = ocr_name_strict_only(heuristic_name_crop(front_pil))

        # Remove noise like **— , “ , ” , out-of-ASCII**
        extracted_name = re.sub(r"[^A-Za-z .]", "", extracted_name)
        extracted_name = re.sub(r"\s{2,}", " ", extracted_name).strip()

        # ---------- Validate fields ----------
        a_val = validate_aadhaar_number(extracted_aadhaar) or "Invalid"
        n_val = validate_name(extracted_name) or "Invalid"
        g_val = validate_gender(extracted_gender) or "Invalid"
        d_val = validate_dob(extracted_dob) or "Invalid"

        # -- Aadhaar validation
        if "Invalid" in a_val:
            results["fraud_score"] += 3
            results["indicators"].append("🔴 HIGH: Invalid Aadhaar number.")
        else:
            results["indicators"].append("✅ LOW: Aadhaar number extracted.")

        # -- Name validation
        if "Invalid" in n_val:
            results["fraud_score"] += 1
            results["indicators"].append(f"🟡 MEDIUM: Name '{extracted_name}' is {n_val}.")
        else:
            results["indicators"].append("✅ LOW: Name format valid.")

        # -- DOB validation
        if "Invalid" in d_val:
            results["fraud_score"] += 2
            results["indicators"].append(f"🔴 HIGH: DOB '{extracted_dob}' invalid.")
        else:
            results["indicators"].append("✅ LOW: DOB format valid.")

        # -- Gender validation
        if "Invalid" in g_val:
            results["fraud_score"] += 1
            results["indicators"].append(f"🟡 MEDIUM: Gender '{extracted_gender}' invalid.")
        else:
            results["indicators"].append("✅ LOW: Gender format valid.")

        # Store final fields
        results["extracted"] = {
            "name": extracted_name,
            "dob": extracted_dob,
            "gender": extracted_gender,
            "aadhaar": extracted_aadhaar
        }

        # ---------- QR Decode ----------
        if do_qr_check:
            qr = decode_secure_qr(img_np_bgr)
            results["qr_data"] = qr
            if "error" not in qr:
                results["indicators"].append("✅ LOW: Secure QR decoded.")
            else:
                results["indicators"].append("⚠️ QR decode failed.")

        # ---------- Final Assessment ----------
        if results["fraud_score"] >= 7:
            results["assessment"] = "HIGH"
        elif results["fraud_score"] >= 3:
            results["assessment"] = "MODERATE"
        else:
            results["assessment"] = "LOW"

        return sanitize_json(results)

    except Exception as e:
        return sanitize_json({
            "error": "PROCESSING_FAILED",
            "message": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": ts
        })


# ---------- Batch ZIP processor ----------
def process_zip_bytes(zip_bytes, model_path=None, do_qr_check=False, device="cpu", max_files=None):
    out = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            imgs = [f for f in z.namelist() if f.lower().endswith((".jpg",".jpeg",".png"))]
            if max_files:
                imgs = imgs[:max_files]

            for name in imgs:
                try:
                    b = z.read(name)
                    res = process_single_image_bytes(b, None, do_qr_check, device)
                    res["filename"] = name
                    out.append(res)
                except Exception as e:
                    out.append({"filename": name, "error": str(e)})

    except Exception as e:
        out.append({"error": str(e)})

    return out
