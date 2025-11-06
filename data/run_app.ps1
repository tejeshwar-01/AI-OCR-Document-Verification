#just type run_app.ps1 directly in powershell to run without delay



Write-Host "🚀 Launching AI-OCR Document Fraud Detection App..." -ForegroundColor Cyan

# 1️⃣ Navigate to project root
cd $PSScriptRoot

# 2️⃣ Allow script execution for this session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# 3️⃣ Activate virtual environment
if (Test-Path "yolov8env/Scripts/Activate.ps1") {
    Write-Host "🔹 Activating Python virtual environment..."
    .\yolov8env\Scripts\Activate.ps1
} else {
    Write-Host "❌ Virtual environment not found! Please run setup_yolov8.ps1 first." -ForegroundColor Red
    exit
}

# 4️⃣ Verify app.py exists
if (-not (Test-Path "app.py")) {
    Write-Host "❌ app.py not found in project root." -ForegroundColor Red
    exit
}

# 5️⃣ Launch Streamlit app
Write-Host "🌐 Running Streamlit web app..." -ForegroundColor Green
streamlit run app.py
