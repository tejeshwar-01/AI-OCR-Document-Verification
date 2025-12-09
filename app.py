import os
import sys
import traceback
from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
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
# LOAD PROCESSOR MODULE
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
# FLASK APP CONFIG
# ------------------------------
app = Flask(
    __name__,
    static_folder="frontend",
    static_url_path=""
)

CORS(app)

# ------------------------------
# NO CACHE (IMPORTANT)
# ------------------------------
@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


# ----------------------------------------------------------
# AUTO REDIRECT LOGIC
# FRONTEND sends cookie/localStorage → JS not python
# So Python cannot read localStorage directly.
#
# FIX:
# We always serve login.html, and JS checks:
#   if(localStorage.user_name) → redirect to dashboard.html
# ----------------------------------------------------------

@app.route("/")
def root():
    """
    Always load login.html first.
    Frontend JS will redirect to dashboard.html if logged in.
    """
    return send_from_directory(FRONTEND, "login.html")


@app.route("/<path:filename>")
def serve_any(filename):
    path = os.path.join(FRONTEND, filename)

    if os.path.exists(path):
        resp = send_from_directory(FRONTEND, filename)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    # fallback → login page
    return send_from_directory(FRONTEND, "login.html")


# ------------------------------
# API — VERIFY SINGLE
# ------------------------------
@app.route("/api/verify_single", methods=["POST"])
def api_single():
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
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ------------------------------
# API — VERIFY BATCH ZIP
# ------------------------------
@app.route("/api/verify_batch", methods=["POST"])
def api_batch():
    if not BACKEND_OK:
        return jsonify({"success": False, "error": "backend_not_loaded"}), 500

    if "zip" not in request.files:
        return jsonify({"success": False, "error": "zip required"}), 400

    try:
        zip_bytes = request.files["zip"].read()
        max_files = request.form.get("max_files")

        results = process_zip_bytes(
            zip_bytes,
            do_qr_check=False,
            max_files=int(max_files) if max_files else None
        )

        return jsonify({"success": True, "results": results})

    except Exception as e:
        print("❌ Batch verify error:", e)
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ------------------------------
# RUN SERVER
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Running on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Allow your frontend origin (replace with your Netlify URL once ready)
CORS(app, origins=["*"])  # replace "*" with "https://your-site.netlify.app" for production

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

# existing routes and app.run or WSGI app object below...
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
