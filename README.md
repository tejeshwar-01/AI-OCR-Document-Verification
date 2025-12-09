📘 Aadhaar Authentication – Aadhaar Card Verification System

A comprehensive web-based solution for automated Aadhaar card verification, fraud detection, OCR extraction, and batch processing with detailed analytics and reporting.

🚀 Features
✔️ Core Verification

Single Aadhaar Verification – Analyze one card with detailed field extraction

Batch Processing – Verify multiple Aadhaar cards at once (ZIP upload)

Automatic Aadhaar Detection – YOLO-based card & field detection

QR Code Verification – Decode Aadhaar QR and cross-validate extracted fields

Multi-Factor Fraud Detection – Consistency checks + risk scoring

✔️ Technical Capabilities

OCR Text Extraction using Tesseract

Photo / Face Detection (YOLO / OpenCV)

Verhoeff Checksum Validation for Aadhaar numbers

Smart Image Processing – denoise, align, enhance

JSON & CSV Export

100% Local Processing (no cloud dependency)

📁 Project Structure (Actual)
aadhaar-fraud-detection-ai/
├── analytics.html
├── about.html
├── contact.html
├── dashboard.html
├── history.html
├── index.html
├── login.html
├── services.html
├── verify-enhanced.html
├── css/
│   └── style.css
├── js/
│   └── script.js
└── backend/
    ├── app.py
    ├── load_model.py
    ├── history.json
    ├── models/
    │   └── best.pt
    └── utils/
        ├── processor.py
        ├── verification_rules.py
        └── ocr_utils.py

🛠️ Technology Stack
🖥️ Frontend

HTML5 / CSS3

JavaScript (Vanilla)

Chart.js (Analytics graphs)

SweetAlert2 (UI alerts)

LocalStorage for verification history

⚙️ Backend

Python (Flask API)

YOLO (Ultralytics) – Aadhaar detection

OpenCV – Image preprocessing & face detection

Tesseract OCR – Text extraction

PyZbar / PyAadhaar – QR decoding

🚀 Quick Start
1️⃣ Prerequisites

Python 3.8+

Tesseract OCR installed

Working browser

YOLO model (best.pt) downloaded

2️⃣ Install the Project
Clone the repository:
git clone <repository-url>
cd aadhaar-fraud-detection-ai

Install Python dependencies:
pip install flask flask-cors ultralytics opencv-python pillow pytesseract pyzbar pyaadhaar

Install Tesseract OCR:
OS	Install
Windows	Download from: https://github.com/UB-Mannheim/tesseract/wiki

Linux	sudo apt-get install tesseract-ocr
macOS	brew install tesseract
Place your YOLO model:
backend/models/best.pt

▶️ Running the Application
Start backend:
cd backend
python app.py


This runs at:

http://localhost:5000

Access frontend:

Open browser → http://localhost:5000

📊 Usage Guide
🔹 Single Verification

Go to Services → Single Verification

Upload front image (required)

Upload back image (optional, for QR check)

Run verification

View:

Extracted fields

Fraud score

Confidence

Risk classification

Download JSON/CSV

🔹 Batch Verification

Upload ZIP containing multiple Aadhaar images

Run processor

Review:

Batch summary

Per-file results

Download combined report

Supported Formats

Images: jpg, jpeg, png, bmp, tiff

Batch ZIP: only images inside

Max upload size: 50MB

🔍 Verification Workflow
1️⃣ Aadhaar Card Detection

YOLO detects Aadhaar region

Text regions identified

Card orientation validated

2️⃣ Field Extraction & Validation

Extract Aadhaar number, name, DOB, gender

Validate:

Verhoeff checksum

DOB formatting

Gender consistency

3️⃣ Fraud Detection

Face/photo detection

OCR vs QR comparison

Heuristic checks

Risk scoring algorithm

4️⃣ Risk Classification
Level	Meaning
LOW	All checks passed
MODERATE	Minor inconsistencies
HIGH	Potential fraud indicators
📋 API Endpoints
Verification
POST /api/verify_single
POST /api/verify_batch

Utility
GET /api/health
GET /

📁 Export Formats
JSON Output Includes:

Timestamp

Extracted fields

OCR confidence

Fraud/risk analysis

QR validation status

CSV Output Includes:

Flattened row for each verification

Summary statistics

Risk level & fraud score

⚙️ Configuration
Model Path

Update in:

backend/app.py

backend/load_model.py

Tesseract Path

Edit:

backend/utils/ocr_utils.py

Custom Logic

Risk rules → verification_rules.py

Fraud scoring → processor.py

UI theme → css/style.css

🛡️ Security & Privacy

No cloud upload

All data processed locally

Temporary files auto-cleaned

No long-term storage unless explicitly implemented

🐛 Troubleshooting
Issue	Fix
Tesseract not found	Add to PATH or set in ocr_utils.py
Model file missing	Ensure best.pt exists in /backend/models/
QR decode fails	Use higher quality back image
Analytics shows 0 history	Check LocalStorage: aadhaar_history
Memory errors	Reduce batch size; process sequentially
📈 Performance

Single Verification: 2–5 seconds

Batch Verification: depends on images

GPU Support: Faster YOLO inference

🧪 Development Tips

Use Incognito Mode to avoid browser extension interference

Restart Flask after backend edits

Use console logs in both browser and terminal for debugging

🔁 Contributing

Fork repo

Create new branch

Add features/fixes

Submit PR with explanation


## Project By

- **M.R.TejeshwarReddy**
- **Contact: tejeshwarreddy2424@gmail.com**

## 📄 License

This project is under MIT license.
