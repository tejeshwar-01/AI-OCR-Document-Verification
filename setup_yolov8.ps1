# type setup_yoloov8.ps1 in powershell to install whole project and run 
# =====================================================
# 🧠 YOLOv8 + GPU Setup Script for Windows (RTX 4050)
# Author: ChatGPT
# =====================================================

Write-Host "`n🚀 Starting YOLOv8 + GPU environment setup..." -ForegroundColor Cyan

# Step 1: Check Python
$pythonVersion = & python --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ Python not found! Please install Python 3.10 (64-bit) from https://www.python.org/downloads/release/python-3109/" -ForegroundColor Red
    exit
}
Write-Host "✅ Python detected: $pythonVersion"

# Step 2: Create virtual environment
if (-Not (Test-Path "yolov8env")) {
    Write-Host "`n📦 Creating virtual environment (yolov8env)..."
    python -m venv yolov8env
}
Write-Host "✅ Virtual environment created."

# Step 3: Activate environment
Write-Host "`n🔄 Activating environment..."
& .\yolov8env\Scripts\activate

# Step 4: Upgrade pip
Write-Host "`n⬆️ Upgrading pip..."
pip install --upgrade pip

# Step 5: Install CUDA PyTorch
Write-Host "`n⚡ Installing CUDA-enabled PyTorch (for RTX 4050)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 6: Install YOLOv8
Write-Host "`n🤖 Installing YOLOv8..."
pip install ultralytics

# Step 7: GPU Verification
Write-Host "`n🔍 Verifying GPU setup..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"

# Step 8: Optional YOLO Test
$weightsPath = "models/yolo_fraud/fraud105/weights/best.pt"
if (Test-Path $weightsPath) {
    Write-Host "`n🧠 Running YOLO test inference..."
    python -c "from ultralytics import YOLO; model = YOLO('$weightsPath'); model.predict(source='data/fraud_dataset/test/images', save=True, conf=0.5)"
    Write-Host "`n✅ Test complete! Check 'runs/detect/predict/' for output images."
} else {
    Write-Host "`n⚠️ Skipping test — model weights not found at $weightsPath" -ForegroundColor Yellow
}

Write-Host "`n🎯 YOLOv8 setup completed successfully!" -ForegroundColor Green
pause
