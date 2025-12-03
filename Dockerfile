# ─────────────────────────────────────────────
# ✅ 1. Use lightweight Python image
FROM python:3.10-slim

# ─────────────────────────────────────────────
# ✅ 2. Install system dependencies required for OCR and YOLO
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libzbar0 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# ✅ 3. Set working directory
WORKDIR /app

# ─────────────────────────────────────────────
# ✅ 4. Copy only requirements first (to leverage Docker caching)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# ✅ 5. Copy the rest of the project files
COPY . .

# ─────────────────────────────────────────────
# ✅ 6. Create runtime directories
RUN mkdir -p backend/models backend/uploads /tmp/uploads

# ─────────────────────────────────────────────
# ✅ 7. Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# ─────────────────────────────────────────────
# ❗ Railway dynamically assigns PORT
# ✔ Do NOT hardcode: EXPOSE 8080
# ✔ EXPOSE $PORT is safe (Docker ignores it)
EXPOSE $PORT

# ─────────────────────────────────────────────
# ⭐ FINAL FIX — Dynamic Port (IMPORTANT)
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300
