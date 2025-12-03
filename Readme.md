# AadhaarVerify - Aadhaar Card Verification System

A comprehensive web-based solution for automated Aadhaar card verification, fraud detection, and batch processing with detailed reporting capabilities.

## 🚀 Features

### Core Verification

- **Single Aadhaar Verification**: Individual card analysis with detailed fraud detection
- **Batch Processing**: Process multiple Aadhaar cards simultaneously
- **Automatic Aadhaar Detection**: Smart detection of valid Aadhaar card images
- **QR Code Verification**: Secure QR code decoding and validation
- **Multi-factor Fraud Detection**: Comprehensive risk assessment

### Technical Capabilities

- **OCR Text Extraction**: Advanced text recognition from Aadhaar cards
- **Face Detection**: Automated face detection on card photos
- **Data Validation**: Verhoeff algorithm validation for Aadhaar numbers
- **Risk Scoring**: Intelligent fraud scoring system
- **Image Processing**: Automated image enhancement and analysis

### Reporting & Export

- **Detailed Reports**: Comprehensive verification results
- **Multiple Formats**: JSON and CSV export options
- **Batch Summaries**: Aggregated results for multiple files
- **Risk Analysis**: Detailed risk breakdown and scoring

## 🛠️ Technology Stack

### Frontend

- **HTML5/CSS3**: Responsive, modern UI with dark theme
- **JavaScript**: Dynamic client-side functionality
- **Fetch API**: Asynchronous server communication

### Backend

- **Python Flask**: RESTful API server
- **YOLO Model**: Object detection for field extraction
- **Tesseract OCR**: Text recognition engine
- **OpenCV**: Image processing and analysis
- **PyAadhaar**: Secure QR code decoding

## 📁 Project Structure

```
aadhaar-verify/
├── frontend/
│   ├── index.html          # Homepage
│   ├── services.html       # Verification services
│   ├── about.html          # About page
│   ├── contact.html        # Contact form
│   ├── css/
│   │   └── style.css       # Main stylesheet
│   └── js/
│       └── script.js       # Client-side functionality
├── backend/
│   ├── app.py              # Flask application
│   ├── utils/
│   │   ├── processor.py    # Main processing logic
│   │   ├── verification_rules.py  # Validation rules
│   │   └── ocr_utils.py    # OCR utilities
│   └── models/
│       └── best.pt         # YOLO model file
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Tesseract OCR
- Modern web browser

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd aadhaar-verify
   ```

2. **Install Python dependencies**

   ```bash
   pip install flask flask-cors ultralytics opencv-python pillow pytesseract pyzbar pyaadhaar
   ```

3. **Install Tesseract OCR**

   - **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`

4. **Place YOLO model**
   - Download the trained YOLO model (`best.pt`)
   - Place it in `backend/models/best.pt`

### Running the Application

1. **Start the backend server**

   ```bash
   python app.py
   ```

   Server will start at `http://localhost:5000`

2. **Access the application**
   - Open your web browser
   - Navigate to `http://localhost:5000`
   - The frontend will be served automatically

## 📊 Usage Guide

### Single Verification

1. Navigate to **Services** page
2. Select **Single Verification** card
3. Upload front Aadhaar image (required)
4. Optionally upload back image for QR verification
5. Enable QR code check if needed
6. Click **Run Verification**
7. View results and download detailed report

### Batch Verification

1. Select **Batch Verification** card
2. Upload ZIP file containing multiple Aadhaar images
3. Click **Run Verification**
4. Review batch summary and individual results
5. Download comprehensive batch report

### Supported File Formats

- **Images**: JPG, JPEG, PNG, BMP, TIFF
- **Batch**: ZIP archives containing images
- **Max Size**: 50MB per upload

## 🔍 Verification Process

The system performs multiple verification steps:

1. **Aadhaar Card Detection**

   - Keyword matching for Aadhaar-specific text
   - Aspect ratio validation
   - Image size verification
   - Confidence scoring

2. **Field Extraction & Validation**

   - Aadhaar number (with Verhoeff checksum)
   - Name validation
   - Date of birth verification
   - Gender validation

3. **Fraud Detection**

   - Face detection on photo
   - Data consistency checks
   - QR code cross-validation
   - Risk scoring algorithm

4. **Risk Assessment**
   - **LOW**: All checks passed
   - **MODERATE**: Minor issues detected
   - **HIGH**: Significant fraud indicators

## 📋 API Endpoints

### Verification Endpoints

- `POST /api/verify_single` - Single Aadhaar verification
- `POST /api/verify_batch` - Batch Aadhaar verification

### Utility Endpoints

- `GET /api/health` - Health check and system status
- `GET /` - Frontend serving

## 📁 Export Formats

### JSON Export

Complete structured data including:

- Verification metadata
- Extracted field data
- Validation results
- Risk assessments
- Processing indicators
- Confidence scores

### CSV Export

Spreadsheet-friendly format with:

- Summary statistics
- Individual file results
- Risk classifications
- Validation status
- Fraud scores

## ⚙️ Configuration

### Environment Setup

- **Model Path**: Update `MODEL_PATH` in `app.py` if needed
- **Tesseract**: Configure path in `ocr_utils.py` for your OS
- **Device**: CPU/GPU selection in processing functions

### Customization

- Modify validation rules in `verification_rules.py`
- Adjust risk scoring in `processor.py`
- Customize UI themes in `style.css`

## 🛡️ Security & Privacy

- Local processing - no data leaves your server
- Temporary file handling
- Secure input validation
- No personal data storage

## 🐛 Troubleshooting

### Common Issues

1. **Tesseract not found**

   - Verify Tesseract installation
   - Check path configuration in `ocr_utils.py`

2. **Model file missing**

   - Ensure `best.pt` is in `backend/models/`
   - Check file permissions

3. **QR decoding errors**

   - Ensure back image is clear and properly oriented
   - Check pyzbar dependencies

4. **Memory issues with large batches**
   - Reduce batch size
   - Increase server resources

### Logs & Debugging

- Check browser console for client errors
- Monitor Flask server logs for backend issues
- Enable debug mode in `app.py` for detailed logging

## 📈 Performance

- **Single Verification**: 2-5 seconds
- **Batch Processing**: Varies with file count and size
- **Memory Usage**: Optimized for typical server environments
- **Concurrency**: Supports multiple simultaneous verifications

## Project By

- **Maddineni Kinshuk**
- **Contact: kinshuk.maddineni@gmail.com**

## 📄 License

This project is under MIT license.
