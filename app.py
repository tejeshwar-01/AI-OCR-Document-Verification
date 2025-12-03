# app.py
import os
import sys
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------
# FIX PYTHON PATH
# ---------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE)
sys.path.append(os.path.join(BASE, "backend"))
sys.path.append(os.path.join(BASE, "backend", "utils"))

# ---------------------------------------------------
# IMPORT PROCESSOR
# ---------------------------------------------------
try:
    from backend.utils.processor import process_single_image_bytes, process_zip_bytes
    BACKEND_OK = True
    print("✅ processor module loaded")
except Exception as e:
    BACKEND_OK = False
    print("❌ Failed to load processor:", e)
    print(traceback.format_exc())

# ---------------------------------------------------
# FLASK SETUP
# ---------------------------------------------------
app = Flask(__name__, static_folder="frontend")
CORS(app)

FRONTEND = os.path.join(BASE, "frontend")

# ------------------------- FRONTEND ROUTES -------------------------

@app.route("/")
def home():
    return send_from_directory(FRONTEND, "index.html")


@app.route("/<path:path>", methods=["GET"])
def serve_static(path):
    file_path = os.path.join(FRONTEND, path)
    if os.path.exists(file_path):
        return send_from_directory(FRONTEND, path)
    return send_from_directory(FRONTEND, "index.html")


# ------------------------- HEALTH CHECK -------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if BACKEND_OK else "error",
        "backend_loaded": BACKEND_OK
    })


# ------------------------- VERIFY SINGLE IMAGE -------------------------

@app.route("/api/verify_single", methods=["POST"])
def verify_single_api():
    if not BACKEND_OK:
        return jsonify({"success": False, "error": "backend_not_loaded"}), 500

    if "front" not in request.files:
        return jsonify({"success": False, "error": "front image required"}), 400

    try:
        img_bytes = request.files["front"].read()
        do_qr = request.form.get("do_qr", "false").lower() in ("1", "true", "yes")

        result = process_single_image_bytes(
            front_bytes=img_bytes,
            back_bytes=None,
            do_qr_check=do_qr,
            device="cpu"
        )

        return jsonify({"success": True, "result": result})

    except Exception as e:
        print("❌ Error verifying single image:", e)
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ------------------------- VERIFY ZIP BATCH -------------------------

@app.route("/api/verify_batch", methods=["POST"])
def verify_batch_api():
    if not BACKEND_OK:
        return jsonify({"success": False, "error": "backend_not_loaded"}), 500

    if "zip" not in request.files:
        return jsonify({"success": False, "error": "zip file required"}), 400

    try:
        zip_bytes = request.files["zip"].read()
        max_files = request.form.get("max_files")

        result = process_zip_bytes(
            zip_bytes,
            do_qr_check=False,
            max_files=int(max_files) if max_files else None
        )

        return jsonify({"success": True, "results": result})

    except Exception as e:
        print("❌ Error verifying batch:", e)
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ------------------------- RUN SERVER -------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n🚀 Aadhaar Verification Server running on http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
import os
from flask import Flask

app = Flask(__name__)

# your routes here...

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
