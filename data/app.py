import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import io

st.set_page_config(page_title="AI Document Fraud Detection", layout="centered")
st.title("📄 AI Document Fraud Detection System")
st.write("Upload a document image (Aadhaar, PAN, Cheque, etc.) to detect fraud patterns using YOLOv8.")

# Load YOLO model
model_path = "models/yolo_fraud/fraud_gpu/weights/best.pt"
model = YOLO(model_path)

# Upload file
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file with a proper extension
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Display image
    image = Image.open(temp_path)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run YOLO detection
    st.info("🔍 Running YOLOv8 detection...")
    results = model.predict(source=temp_path, conf=0.5, save=False)

    # Display annotated image
    res_plot = results[0].plot()
    st.image(res_plot, caption="Detection Results", use_container_width=True)

    # Show detected labels
    detected_classes = [model.names[int(c)] for c in results[0].boxes.cls]
    st.success(f"✅ Detected: {set(detected_classes)}")

    # Clean up
    os.unlink(temp_path)
