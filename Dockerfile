FROM python:3.10-slim

# ----------------------------------------------------
# 1. Prevent Matplotlib from building font cache
# ----------------------------------------------------
ENV MPLCONFIGDIR=/tmp/matplotlib

# ----------------------------------------------------
# 2. Install system dependencies (FULL TESSERACT + OPENGL)
# ----------------------------------------------------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libzbar0 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------
# 3. App directory
# ----------------------------------------------------
WORKDIR /app

# ----------------------------------------------------
# 4. Install Python dependencies
# ----------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# ----------------------------------------------------
# 5. Environment variables
# ----------------------------------------------------
ENV FLASK_ENV=production
ENV GUNICORN_WORKERS=1
ENV PYTHONUNBUFFERED=1

# Increase Gunicorn timeout for OCR+YOLO
ENV GUNICORN_TIMEOUT=300

# ----------------------------------------------------
# 6. Expose port
# ----------------------------------------------------
ENV PORT=8000
EXPOSE 8000

# ----------------------------------------------------
# 7. Run Gunicorn with safe production settings
# ----------------------------------------------------
CMD ["gunicorn",
     "--bind", "0.0.0.0:8000",
     "--workers", "1",
     "--timeout", "300",
     "--log-level", "debug",
     "app:app"]
