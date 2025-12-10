FROM python:3.10-slim

# ----------------------------------------------------
# Prevent Matplotlib heavy font cache
# ----------------------------------------------------
ENV MPLCONFIGDIR=/tmp/matplotlib

# ----------------------------------------------------
# Install system dependencies for OCR + CV
# ----------------------------------------------------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-tam \
    libtesseract-dev \
    libzbar0 \
    libgl1 \
    ffmpeg \
    fonts-samyak-tamil \
    fonts-samyak-devanagari \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------
# Working directory
# ----------------------------------------------------
WORKDIR /app

# ----------------------------------------------------
# Install Python dependencies
# ----------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------
# Copy project
# ----------------------------------------------------
COPY . .

# ----------------------------------------------------
# Environment
# ----------------------------------------------------
ENV FLASK_ENV=production
ENV GUNICORN_WORKERS=1
ENV GUNICORN_TIMEOUT=300
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# ----------------------------------------------------
# Gunicorn CMD
# ----------------------------------------------------
CMD ["gunicorn", "app:app", \
"--bind=0.0.0.0:8000", \
"--workers=1", \
"--timeout=300", \
"--log-level=debug"]
