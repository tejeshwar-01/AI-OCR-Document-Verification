# AI-OCR-Document-Verification
An AI-powered Optical Character Recognition (OCR) and document verification system using Azure Document Intelligence and Azure OpenAI. This project automatically extracts and validates data from scanned Aadhaar cards and other ID documents.
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Full Setup & Run Guide (PowerShell)

This guide fully automates the setup, training, and running of your **AI-OCR Document Fraud Detection System** using YOLOv8 and Streamlit — all from Windows PowerShell.

---

## ⚙️ 1. What this project does

* Detects fraud patterns (forged Aadhaar, PAN, cheque, etc.) using a trained **YOLOv8** model.
* Uses a YOLO-formatted dataset with `train`, `valid`, `test` splits.
* Offers a **Streamlit web app** for uploading and detecting document fraud visually.

---

## 🧩 2. Folder structure

Your project folder should look like this:

```
AI-OCR-Document-Verification-main/
│
├── app.py                    # Streamlit web app
├── data/
│   └── fraud_dataset/
│       ├── train/images & labels
│       ├── valid/images & labels
│       └── test/images & labels
│
├── models/
│   └── yolo_fraud/           # trained weights and logs stored here
│
├── yolov8env/                # virtual environment (auto-created)
│
├── setup_yolov8.ps1          # setup + train + launch automation script (you will create it)
└── README.md                 # this guide
```

---

## 🚀 3. Create PowerShell setup script — `setup_yolov8.ps1`

Create a new file named **`setup_yolov8.ps1`** in your project root and paste the following:

```powershell
Write-Host "🚀 Starting full AI-OCR setup..." -ForegroundColor Cyan

# 1️⃣ Ensure we're in project root
cd $PSScriptRoot

# 2️⃣ Allow script execution for this session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# 3️⃣ Create venv if missing
if (-not (Test-Path "yolov8env")) {
    Write-Host "📦 Creating Python virtual environment (yolov8env)..."
    python -m venv yolov8env
}

# 4️⃣ Activate venv
Write-Host "🔹 Activating environment..."
.\yolov8env\Scripts\Activate.ps1

# 5️⃣ Upgrade pip
Write-Host "⬆️  Upgrading pip..."
pip install -U pip

# 6️⃣ Install dependencies
Write-Host "📥 Installing required libraries (Ultralytics, Torch, Streamlit)..."
pip install ultralytics streamlit torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 7️⃣ Verify GPU access
Write-Host "🧠 Checking CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# 8️⃣ Validate dataset presence
if (-not (Test-Path "data/fraud_dataset/train/images")) {
    Write-Host "❌ Dataset missing! Please extract your YOLO dataset into data/fraud_dataset/." -ForegroundColor Red
    exit
} else {
    Write-Host "✅ Dataset found."
}

# 9️⃣ Train YOLOv8 model (optional, comment out if you already have weights)
Write-Host "⚙️ Training YOLOv8 model for 5 epochs..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='data/fraud_dataset.yaml', epochs=5, project='models/yolo_fraud', name='fraud_gpu')"

# 🔟 Launch Streamlit app
Write-Host "🌐 Launching Streamlit app..."
streamlit run app.py
```

---

## ▶️ 4. Run the setup script

From PowerShell (in your project root):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\setup_yolov8.ps1
```

This script will:

1. Create & activate your virtual environment
2. Install dependencies (Ultralytics, Torch + CUDA, Streamlit)
3. Check GPU availability
4. Verify dataset folder
5. Train YOLOv8 for 5 epochs (if not already done)
6. Launch the Streamlit app automatically

---

## 🧠 5. How the app works internally

1. **YOLOv8 Training** → Uses `ultralytics` to fine-tune the base model on your dataset.
2. **Streamlit UI** → Provides an upload interface to test new document images.
3. **Model Prediction** → `model.predict()` reads the uploaded image and detects fraud types.
4. **Display Output** → Streamlit renders detection boxes and class labels in real time.

---

## 🧾 6. Dataset YAML reference

Example: `data/fraud_dataset.yaml`

```yaml
train: data/fraud_dataset/train/images
val: data/fraud_dataset/valid/images
test: data/fraud_dataset/test/images

nc: 4
names: ["aadhaar", "pan", "cheque", "other"]
```

Ensure paths match your folder layout exactly.

---

## 🧰 7. Manual command references (if running without script)

### Activate venv

```powershell
.\yolov8env\Scripts\Activate.ps1
```

### Train YOLOv8 manually

```powershell
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='data/fraud_dataset.yaml', epochs=5, project='models/yolo_fraud', name='fraud_gpu')"
```

### Run inference on test set

```powershell
python -c "from ultralytics import YOLO; m=YOLO('models/yolo_fraud/fraud_gpu/weights/best.pt'); m.predict(source='data/fraud_dataset/test/images', save=True, conf=0.5)"
```

### Launch app manually

```powershell
streamlit run app.py
```

---

## 🏁 8. Outputs & logs

* YOLO training weights → `models/yolo_fraud/fraud_gpu/weights/best.pt`
* Streamlit detections → shown live in browser (saved optionally under `/runs/detect/predict`)
* Logs → visible in PowerShell and Streamlit console

---

## ✅ 9. End result

After successful setup and execution, your system will:

* Train YOLOv8 on your fraud dataset
* Detect Aadhaar, PAN, cheque, and other fraud types
* Display bounding boxes and class predictions interactively in Streamlit

| Parameter | Description |
|------------|-------------|
| 🧠 **Model Used** | YOLOv8n (fine-tuned on fraud detection dataset) |
| 📦 **Weights Path** | `models/yolo_fraud/fraud_gpu/weights/best.pt` |
| ⚙️ **Classes** | `['aadhaar', 'pan', 'cheque', 'other']` |
