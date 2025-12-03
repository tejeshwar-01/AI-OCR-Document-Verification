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

# Install Python dependencies (no cache to keep image small)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# ✅ 5. Copy the rest of the project files
COPY . .

# ─────────────────────────────────────────────
# ✅ 6. Create runtime directories
RUN mkdir -p backend/models backend/uploads /tmp/uploads

# ─────────────────────────────────────────────
# ✅ 7. Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# ─────────────────────────────────────────────
# ✅ 8. Expose Railway default port
EXPOSE 8080

# ─────────────────────────────────────────────
# ✅ 9. Start the app with Gunicorn (using the correct port)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--timeout", "300"]
