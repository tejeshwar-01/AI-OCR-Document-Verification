import os
import sys
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(BASE, "frontend")
BACKEND = os.path.join(BASE, "backend")

sys.path.append(BACKEND)
sys.path.append(os.path.join(BACKEND, "utils"))

# Try loading backend modules
try:
    from backend.utils.processor import process_single_image_bytes, process_zip_bytes
    BACKEND_OK = True
except Exception as e:
    print("❌ Failed to load processor:", e)
    BACKEND_OK = False


# ------------------------------------------------------
# CREATE ONLY ONE FLASK APP
# ------------------------------------------------------
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ------------------------------------------------------
# HEALTH CHECK  ✅
# ------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ------------------------------------------------------
# SERVE FRONTEND  ✅
# ------------------------------------------------------
@app.route("/")
def root():
    return send_from_directory(FRONTEND, "index.html")

@app.route("/<path:filename>")
def serve_any(filename):
    filepath = os.path.join(FRONTEND, filename)
    if os.path.exists(filepath):
        return send_from_directory(FRONTEND, filename)
    return send_from_directory(FRONTEND, "index.html")


# ------------------------------------------------------
# VERIFY SINGLE  ✅
# ------------------------------------------------------
@app.route("/api/verify_single", methods=["POST"])
def verify_single():
    if not BACKEND_OK:
        return jsonify({"error": "backend-not-loaded"}), 500

    if "front" not in request.files:
        return jsonify({"error": "front image missing"}), 400

    try:
        front_bytes = request.files["front"].read()
        result = process_single_image_bytes(front_bytes, None, False, "cpu")
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ------------------------------------------------------
# VERIFY BATCH ZIP  ✅
# ------------------------------------------------------
@app.route("/api/verify_batch", methods=["POST"])
def verify_batch():
    if not BACKEND_OK:
        return jsonify({"error": "backend-not-loaded"}), 500

    if "zip" not in request.files:
        return jsonify({"error": "zip file missing"}), 400

    try:
        zip_bytes = request.files["zip"].read()
        results = process_zip_bytes(zip_bytes, False, None)
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ------------------------------------------------------
# RUN APP
# ------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
