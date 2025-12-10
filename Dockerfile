FROM python:3.10-slim

# --------------------------------------------------------
# Install system dependencies + OCR languages + correct fonts
# --------------------------------------------------------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-tam \
    libtesseract-dev \
    libzbar0 \
    libgl1 \
    ffmpeg \
    fonts-lohit-tamil \
    fonts-lohit-devanagari \
    fonts-tamlproj \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------
# Setup work directory
# --------------------------------------------------------
WORKDIR /app

# --------------------------------------------------------
# Install Python dependencies
# --------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------
# Copy project files
# --------------------------------------------------------
COPY . .

ENV PORT=8000
EXPOSE 8000

# --------------------------------------------------------
# Start Gunicorn server
# --------------------------------------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "300", "app:app"]
