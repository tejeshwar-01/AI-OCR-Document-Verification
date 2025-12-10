# ============================================================
#  processor.py  — UNIVERSAL AADHAAR OCR (Tesseract-only final)
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

# ---------- Dependency probes ----------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# We choose Tesseract-only mode as requested
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

# QR + pyaadhaar
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    from pyaadhaar.utils import isSecureQr
    from pyaadhaar.decode import AadhaarSecureQr
    PYAADHAAR_AVAILABLE = True
except Exception:
    PYAADHAAR_AVAILABLE = False

# ============================================================
#  Sanitizer (JSON-safe)
# ============================================================
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

# ============================================================
#  Basic OCR helpers (Tesseract)
# ============================================================
def pytesseract_image_to_text(pil_image):
    if not PYTESSERACT_AVAILABLE:
        return ""
    try:
        arr = np.array(preprocess_for_ocr_full(pil_image) if 'preprocess_for_ocr_full' in globals() else pil_image)
        pil = pil_image if isinstance(pil_image, Image.Image) else Image.fromarray(arr)
        text = pytesseract.image_to_string(pil)
        return text.lower().strip()
    except Exception:
        try:
            return pytesseract.image_to_string(pil_image).lower().strip()
        except Exception:
            return ""

# ============================================================
#  Simple strict OCR helpers for name/dob/gender
# ============================================================
def clean_name_text(txt):
    txt = re.sub(r"[^A-Za-z\s.]", "", txt)
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt.strip()

def ocr_name_strict_only(pil_img):
    if not PYTESSERACT_AVAILABLE:
        return ""
    try:
        img = np.array(pil_img.convert("RGB"))[:, :, ::-1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
        proc = Image.fromarray(gray)
        config = "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz. "
        try:
            txt = pytesseract.image_to_string(proc, config=config)
        except Exception:
            txt = pytesseract.image_to_string(proc)
        return clean_name_text(txt)
    except Exception:
        try:
            txt = pytesseract.image_to_string(pil_img)
            return clean_name_text(txt)
        except Exception:
            return ""

def ocr_dob_strict(pil_img):
    if not PYTESSERACT_AVAILABLE:
        return ""
    try:
        img = np.array(pil_img.convert("RGB"))[:, :, ::-1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        proc_pil = Image.fromarray(th)
        config = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/-."
        try:
            txt = pytesseract.image_to_string(proc_pil, config=config)
        except Exception:
            txt = pytesseract.image_to_string(proc_pil)
        txt = txt.replace(".", "/").replace("-", "/")
        txt = re.sub(r"[^\d/]", "", txt)
        m = re.search(r'(\d{2})/?(\d{2})/?(\d{4})', txt)
        if m:
            return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
        return txt
    except Exception:
        return ""

def ocr_gender_strict(pil_img):
    if not PYTESSERACT_AVAILABLE:
        return ""
    try:
        proc = np.array(pil_img.convert("RGB"))[:, :, ::-1]
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        proc_pil = Image.fromarray(th)
        config = "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        try:
            txt = pytesseract.image_to_string(proc_pil, config=config).lower()
        except Exception:
            txt = pytesseract.image_to_string(proc_pil).lower()
        if "male" in txt:
            return "Male"
        if "female" in txt:
            return "Female"
        letters = re.sub(r"[^a-z]", "", txt)
        if letters == "m":
            return "Male"
        if letters == "f":
            return "Female"
        return ""
    except Exception:
        return ""

# ============================================================
#  Heuristic fixed-region name extractor (kept as fallback)
# ============================================================
def extract_universal_name(full_pil):
    W, H = full_pil.size
    # Old region
    old_crop = full_pil.crop((int(W*0.28), int(H*0.16), int(W*0.95), int(H*0.36)))
    new_crop = full_pil.crop((int(W*0.05), int(H*0.28), int(W*0.90), int(H*0.48)))
    pdf_crop = full_pil.crop((int(W*0.05), int(H*0.10), int(W*0.70), int(H*0.28)))
    fallback_crop = full_pil.crop((int(W*0.24), int(H*0.18), int(W*0.95), int(H*0.50)))
    names = []
    for crop in [old_crop, new_crop, pdf_crop, fallback_crop]:
        try:
            txt = pytesseract.image_to_string(crop, config="--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz. ")
        except Exception:
            txt = pytesseract.image_to_string(crop)
        txt = re.sub(r"[^A-Za-z .]", "", txt)
        txt = re.sub(r"\s{2,}", " ", txt).strip()
        if len(txt.split()) >= 2:
            names.append(txt)
    if names:
        return max(names, key=len)
    return ""

# ============================================================
#  Robust line-based extractor v2 (preferred)
# ============================================================
def extract_universal_name_v2(full_pil):
    """
    Robust name extractor:
    - deskew using pytesseract OSD if available
    - CLAHE + denoise
    - adaptive threshold + morphological closing to isolate text lines
    - find contours, OCR each candidate line
    - skip numeric-heavy lines, pick best name-like line
    """
    try:
        W, H = full_pil.size

        # Deskew using OSD if possible
        try:
            if PYTESSERACT_AVAILABLE:
                osd = pytesseract.image_to_osd(full_pil)
                rot_m = re.search(r'Rotate: (\d+)', osd)
                if rot_m:
                    angle = int(rot_m.group(1))
                    if angle != 0:
                        full_pil = full_pil.rotate(-angle, expand=True)
        except Exception:
            pass

        # Convert to OpenCV BGR
        img = np.array(full_pil.convert("RGB"))[:, :, ::-1]
        orig_h, orig_w = img.shape[:2]

        # Enhance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.medianBlur(gray, 3)

        # Adaptive threshold and invert so text is white
        th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,35,11)
        th = 255 - th

        # Morphological close to join characters into lines
        kernel_w = max(10, orig_w // 60)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x,y,wc,hc = cv2.boundingRect(cnt)
            # Filter by reasonable size
            if wc < orig_w * 0.07 or hc < 10:
                continue
            if hc > orig_h * 0.6:
                continue
            boxes.append((x,y,wc,hc))

        if not boxes:
            # fallback to template method
            return extract_universal_name(full_pil)

        # Sort top->bottom
        boxes = sorted(boxes, key=lambda b: b[1])

        candidates = []
        for (x,y,wc,hc) in boxes:
            padx = int(wc * 0.05)
            pady = int(hc * 0.15)
            x1 = max(0, x - padx)
            y1 = max(0, y - pady)
            x2 = min(img.shape[1], x + wc + padx)
            y2 = min(img.shape[0], y + hc + pady)
            crop_cv = img[y1:y2, x1:x2]
            try:
                crop_pil = Image.fromarray(cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB))
            except Exception:
                continue

            # Quick check: skip numeric-heavy lines
            sample_text = ""
            try:
                if PYTESSERACT_AVAILABLE:
                    sample_text = pytesseract.image_to_string(crop_pil, config="--psm 7").strip()
            except Exception:
                sample_text = ""
            digits = len(re.findall(r"\d", sample_text))
            total_chars = max(1, len(sample_text))
            digit_ratio = digits / total_chars
            if digit_ratio > 0.35:
                continue

            # OCR line (prefer strict name config)
            ocr_text = ""
            try:
                ocr_text = pytesseract.image_to_string(crop_pil, config="--psm 7 --oem 3")
            except Exception:
                try:
                    ocr_text = pytesseract.image_to_string(crop_pil)
                except Exception:
                    ocr_text = ""

            # Allow Devanagari/Tamil characters as well; keep broader charset initially
            clean = re.sub(r"[^A-Za-z\u0900-\u097F\u0B80-\u0BFF\s.]", "", ocr_text)
            clean = re.sub(r"\s{2,}", " ", clean).strip()

            # Heuristics: length & words
            words = clean.split()
            letter_count = len(re.sub(r"[^A-Za-z\u0900-\u097F\u0B80-\u0BFF]", "", clean))
            if 2 <= len(words) <= 6 and letter_count >= 4:
                ascii_letters = len(re.findall(r"[A-Za-z]", clean))
                letters_total = max(1, letter_count)
                ascii_ratio = ascii_letters / letters_total
                candidates.append({
                    "text": clean,
                    "ascii_ratio": ascii_ratio,
                    "word_count": len(words),
                    "length": len(clean)
                })

        if not candidates:
            return extract_universal_name(full_pil)

        # Prefer highest ascii_ratio then longest length
        candidates = sorted(candidates, key=lambda c: (c["ascii_ratio"], c["length"]), reverse=True)
        best = candidates[0]["text"]

        # If candidate is mostly non-latin, but a latin substring exists, prefer latin substring of 2+ words
        if candidates[0]["ascii_ratio"] < 0.5:
            latin_parts = re.findall(r"[A-Za-z\s.]{3,}", best)
            if latin_parts:
                best_latin = max((p.strip() for p in latin_parts), key=len)
                if len(best_latin.split()) >= 2:
                    best = best_latin

        # Final cleanup: keep latin letters only for output (user likely expects romanized), but retain if only native
        # Try to pick romanized substring if exists
        latin_subs = re.findall(r"[A-Za-z\s.]{2,}", best)
        if latin_subs:
            chosen = max(latin_subs, key=len).strip()
            if len(chosen.split()) >= 2:
                best = chosen

        # Normalize
        best_ascii_clean = re.sub(r"[^A-Za-z .]", "", best)
        best_final = re.sub(r"\s{2,}", " ", best_ascii_clean).strip()
        if best_final and len(best_final.split()) >= 2:
            return best_final

        # fallback: return best as-is (possibly native script)
        return best.strip()
    except Exception:
        try:
            return extract_universal_name(full_pil)
        except Exception:
            return ""

# ============================================================
#  Aadhaar heuristic detection
# ============================================================
def is_aadhaar_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        max_dim = 1400
        w, h = image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            w, h = image.size

        np_img = np.array(image)
        aspect_ratio = (w / h) if h else 0.0
        aspect_ok = 1.1 <= aspect_ratio <= 2.9
        size_ok = min(w, h) >= 140

        sample_h = min(80, max(10, h // 8))
        avg_top = np.mean(np_img[:sample_h, :, :], axis=(0, 1))
        orange_hint = bool(avg_top[0] > avg_top[1] * 0.95 and avg_top[0] > avg_top[2] * 0.9)

        txt = ""
        if PYTESSERACT_AVAILABLE:
            txt = pytesseract_image_to_text(image)
        txt = txt or ""
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
            "detected_text_snippet": txt[:300]
        }
        return bool(is_aadhaar), int(confidence), details
    except Exception as e:
        return False, 0, {"error": str(e)}

# ============================================================
#  OCR helper for arbitrary crops
# ============================================================
def ocr_text_for_label(crop_pil, label=""):
    if PYTESSERACT_AVAILABLE:
        try:
            pre = preprocess_for_ocr(crop_pil) if 'preprocess_for_ocr' in globals() else crop_pil
            txt = pytesseract_image_to_text(pre) or ""
            return txt.strip()
        except Exception:
            try:
                return pytesseract_image_to_text(crop_pil)
            except:
                return ""
    return ""

# ============================================================
#  Regex helpers
# ============================================================
AADHAAR_RE = re.compile(r'(\d{4}\s?\d{4}\s?\d{4})')
DOB_RE = re.compile(r'(\d{2}[\/\-\.\s]\d{2}[\/\-\.\s]\d{4})')

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
    m2 = re.search(r'(\d{2}\s?\d{2}\s?\d{4})', txt)
    if m2:
        s = m2.group(1)
        return f"{s[0:2]}/{s[2:4]}/{s[4:8]}"
    return ""

def guess_name_from_text(txt):
    if not txt:
        return ""
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    candidates = []
    for ln in lines[:30]:
        if any(c.isalpha() for c in ln) and 3 < len(ln) < 80:
            low = ln.lower()
            if any(skip in low for skip in ['dob', 'date of birth', 'male', 'female', 'government', 'aadhaar', 'uidai']):
                continue
            candidates.append(ln)
    for c in candidates:
        if c.isupper() and len(c.split()) <= 6:
            return c
    return candidates[0] if candidates else ""

# ============================================================
#  Main processing logic
# ============================================================
def process_single_image_bytes(front_bytes, back_bytes=None, do_qr_check=False, device="cpu"):
    ts = datetime.datetime.now().isoformat()
    try:
        is_aadhaar, aadhaar_confidence, aadhaar_details = is_aadhaar_image(front_bytes)
        if not is_aadhaar:
            return sanitize_json({
                "error": "NOT_AADHAAR",
                "message": "The uploaded image does not appear to be an Aadhaar card",
                "aadhaar_verification": aadhaar_details,
                "confidence_score": int(aadhaar_confidence),
                "timestamp": ts,
                "assessment": "INVALID_INPUT"
            })

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

        # YOLO field detection (optional)
        yolo_model = None
        if YOLO_AVAILABLE:
            try:
                yolo_model = YOLO(os.getenv("YOLO_MODEL_PATH", ""))
            except Exception:
                yolo_model = None

        fields = {}
        if yolo_model is None:
            results["indicators"].append("⚪ INFO: Field detection (YOLO) disabled in this build.")
        else:
            try:
                preds = yolo_model(img_np_rgb, device=device)
                boxes = getattr(preds[0], "boxes", None)
                names = getattr(preds[0], "names", None) or {}
                if boxes is not None:
                    for b in boxes:
                        try:
                            cls = int(b.cls.cpu().numpy()) if hasattr(b, "cls") else int(getattr(b, "cls", 0))
                        except Exception:
                            cls = int(getattr(b, "cls", 0))
                        label = str(names.get(cls, cls)) if names is not None else str(cls)
                        xy = getattr(b, "xyxy", None)
                        coords = None
                        if xy is None:
                            coords = b.xyxy[0] if hasattr(b, "xyxy") else None
                        else:
                            coords = xy[0]
                        if coords is not None:
                            x1, y1, x2, y2 = map(int, coords[:4])
                            h, w = img_np_rgb.shape[:2]
                            x1, x2 = max(0, x1), min(w, x2)
                            y1, y2 = max(0, y1), min(h, y2)
                            crop = front_image_pil.crop((x1, y1, x2, y2))
                            norm_label = str(label).lower()
                            if "name" in norm_label:
                                fields["name"] = crop
                            elif "dob" in norm_label or "date" in norm_label:
                                fields["dob"] = crop
                            elif "gender" in norm_label:
                                fields["gender"] = crop
                            elif "aadhaar" in norm_label or "uid" in norm_label or "number" in norm_label:
                                fields["aadhaar"] = crop
                            else:
                                fields[label] = crop
                if fields:
                    results["indicators"].append("✅ LOW: YOLO field detection ran and returned crops.")
                else:
                    results["indicators"].append("⚪ INFO: YOLO ran but no labelled fields detected.")
            except Exception as e:
                results["indicators"].append(f"⚠️ YOLO detection error: {str(e)}")

        # Full-image OCR (for heuristics & fallback)
        full_text = ""
        if PYTESSERACT_AVAILABLE:
            try:
                full_text = pytesseract_image_to_text(front_image_pil)
                results["ocr_data"]["full_text"] = full_text
            except Exception:
                full_text = ""
        else:
            results["ocr_data"]["full_text"] = ""

        # ---------- Extraction logic ----------
        extracted_name = ""
        extracted_gender = ""
        extracted_dob = ""
        extracted_aadhaar = ""

        # 1) If YOLO gave crops, OCR those specific crops.
        if fields.get("aadhaar"):
            extracted_aadhaar = ocr_text_for_label(fields["aadhaar"], "aadhaar")

        if fields.get("name"):
            try:
                extracted_name = ocr_name_strict_only(fields["name"])
            except Exception:
                extracted_name = ocr_text_for_label(fields["name"], "name")

        if fields.get("dob"):
            try:
                extracted_dob = ocr_dob_strict(fields["dob"])
            except Exception:
                extracted_dob = ocr_text_for_label(fields["dob"], "dob")

        if fields.get("gender"):
            try:
                extracted_gender = ocr_gender_strict(fields["gender"])
            except Exception:
                extracted_gender = ocr_text_for_label(fields["gender"], "gender")

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

        # NAME extraction: use robust v2 extractor first
        try:
            if not extracted_name:
                extracted_name = extract_universal_name_v2(front_image_pil)
        except Exception:
            pass

        # fallback to full text guess
        if not extracted_name:
            guess = guess_name_from_text(results["ocr_data"].get("full_text", "") or full_text)
            extracted_name = guess or ""

        # Additional heuristic fallback with fixed region
        if not extracted_name or len(extracted_name) < 3:
            try:
                alt_name = extract_universal_name(front_image_pil)
                if alt_name:
                    extracted_name = alt_name
            except Exception:
                pass

        # 3) Clean & normalize
        if extracted_aadhaar:
            extracted_aadhaar = re.sub(r'[^0-9]', '', extracted_aadhaar)
        if extracted_name:
            extracted_name = correct_common_ocr_errors(extracted_name).strip()
        if extracted_dob:
            extracted_dob = correct_common_ocr_errors(extracted_dob).strip()

        # 4) Validation & scoring
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

        # 5) QR verification (optional)
        if do_qr_check:
            try:
                qr_data = decode_secure_qr(img_np_bgr) if 'decode_secure_qr' in globals() else {}
                results["qr_data"] = qr_data
                if qr_data and "error" not in qr_data:
                    results["indicators"].append("✅ LOW: Secure QR Code decoded successfully.")
                else:
                    if qr_data:
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

# ============================================================
#  Batch ZIP processing
# ============================================================
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
# ============================================================
# End of file
# ============================================================
