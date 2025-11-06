Write-Host "🚀 Starting full AI-OCR setup..." -ForegroundColor Cyan

# 1️⃣ Ensure script runs from project root
cd $PSScriptRoot

# 2️⃣ Allow script execution for this session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# 3️⃣ Create virtual environment if it doesn't exist
if (-not (Test-Path "yolov8env")) {
    Write-Host "📦 Creating Python virtual environment (yolov8env)..."
    python -m venv yolov8env
} else {
    Write-Host "✅ Virtual environment already exists."
}

# 4️⃣ Activate environment
Write-Host "🔹 Activating environment..."
.\yolov8env\Scripts\Activate.ps1

# 5️⃣ Upgrade pip
Write-Host "⬆️  Upgrading pip..."
pip install -U pip

# 6️⃣ Install required dependencies
Write-Host "📥 Installing required libraries (Ultralytics, Torch, Streamlit)..."
pip install ultralytics streamlit torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 7️⃣ Verify GPU access
Write-Host "🧠 Checking CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# 8️⃣ Validate dataset presence
if (-not (Test-Path "data/fraud_dataset/train/images")) {
    Write-Host "❌ Dataset not found! Please extract YOLO dataset under data/fraud_dataset/." -ForegroundColor Red
    exit
} else {
    Write-Host "✅ Dataset found and ready."
}

# 9️⃣ Run YOLOv8 training (optional, can be commented out if already trained)
Write-Host "⚙️ Training YOLOv8 model for 5 epochs..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='data/fraud_dataset.yaml', epochs=5, project='models/yolo_fraud', name='fraud_gpu')"

# 🔟 Launch the Streamlit web app
Write-Host "🌐 Launching Streamlit app in browser..."
streamlit run app.py
