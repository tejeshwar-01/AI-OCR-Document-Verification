FROM python:3.10-slim

# ----------------------------------------------------
# 1. Prevent Matplotlib from building heavy font cache
# ----------------------------------------------------
ENV MPLCONFIGDIR=/tmp/matplotlib

# ----------------------------------------------------
# 2. Install system dependencies for OCR + CV
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
# 3. Set working directory
# ----------------------------------------------------
WORKDIR /app

# ----------------------------------------------------
# 4. Install Python dependencies
# ----------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------
# 5. Copy project files
# ----------------------------------------------------
COPY . .

# ----------------------------------------------------
# 6. Environment configuration
# ----------------------------------------------------
ENV FLASK_ENV=production
ENV GUNICORN_WORKERS=1
ENV GUNICORN_TIMEOUT=300
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# ----------------------------------------------------
# 7. Correct Gunicorn CMD (NO indentation errors)
# ----------------------------------------------------
CMD ["gunicorn", "app:app", \
"--bind=0.0.0.0:8000", \
"--workers=1", \
"--timeout=300", \
"--log-level=debug"]
