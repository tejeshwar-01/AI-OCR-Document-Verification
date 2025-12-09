import os
import sys
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ------------------------------
# PATH FIX
# ------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(BASE, "frontend")
BACKEND = os.path.join(BASE, "backend")

sys.path.append(BASE)
sys.path.append(BACKEND)
sys.path.append(os.path.join(BACKEND, "utils"))

# ------------------------------
# LOAD PROCESSOR
# ------------------------------
try:
    from backend.utils.processor import process_single_image_bytes, process_zip_bytes
    BACKEND_OK = True
    print("✅ processor loaded")
except Exception as e:
    BACKEND_OK = False
    print("❌ processor load error:", e)
    print(traceback.format_exc())

# ------------------------------
# FLASK APP
# ------------------------------
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app, origins="*")

# ------------------------------
# HEALTH CHECK
# ------------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


# ------------------------------
# SERVE FRONTEND
# ------------------------------
@app.route("/")
def root():
    return send_from_directory("frontend", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    full = os.path.join("frontend", path)
    if os.path.exists(full):
        return send_from_directory("frontend", path)
    return send_from_directory("frontend", "index.html")


# ------------------------------
# VERIFY SINGLE
# ------------------------------
@app.route("/api/verify_single", methods=["POST"])
def verify_single():
    if not BACKEND_OK:
        return jsonify({"success": False, "error": "backend_not_loaded"}), 500

    if "front" not in request.files:
        return jsonify({"success": False, "error": "front image required"}), 400

    try:
        front_bytes = request.files["front"].read()

        qr_flag = request.form.get("qr", "").lower() in ("1", "true", "yes")

        result = process_single_image_bytes(
            front_bytes=front_bytes,
            back_bytes=None,
            do_qr_check=qr_flag,
            device="cpu"
        )

        return jsonify({"success": True, "result": result})

    except Exception as e:
        print("❌ Single verify error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ------------------------------
# VERIFY BATCH
# ------------------------------
@app.route("/api/verify_batch", methods=["POST"])
def verify_batch():
    if not BACKEND_OK:
        return jsonify({"success": False, "error": "backend_not_loaded"}), 500

    if "zip" not in request.files:
        return jsonify({"success": False, "error": "zip required"}), 400

    try:
        zip_bytes = request.files["zip"].read()

        results = process_zip_bytes(zip_bytes, do_qr_check=False)

        return jsonify({"success": True, "results": results})

    except Exception as e:
        print("❌ Batch verify error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print("Running on:", port)
    app.run(host="0.0.0.0", port=port)
