"""
FastAPI backend for the Advanced AI Medical Intelligence Platform.

Endpoints:
    POST /predict          -> upload image, get prediction + Grad-CAM + LLM report
    GET  /history          -> list past predictions
    GET  /history/{id}     -> get a single prediction record
    GET  /gradcam/{fname}  -> serve a saved Grad-CAM image
    GET  /health           -> health check
"""

import os
import io
import uuid
import sys
from dotenv import load_dotenv
load_dotenv()
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.gradcam import GradCAM, overlay_heatmap, estimate_region
from model.train import build_model, CLASS_NAMES
from db.database import init_db, get_db, Prediction
from api.llm_report import generate_report

MODEL_PATH = os.getenv("MODEL_PATH", "saved_models/best_model.pt")
GRADCAM_DIR = os.getenv("GRADCAM_DIR", "static/gradcam")
IMG_SIZE = 224

os.makedirs(GRADCAM_DIR, exist_ok=True)

app = FastAPI(
    title="Advanced AI Medical Intelligence Platform",
    description="Deep learning based chest X-ray screening with Grad-CAM explainability and LLM-generated reports.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
classes = CLASS_NAMES

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.on_event("startup")
def startup():
    global model, classes
    init_db()

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        classes = checkpoint.get("classes", CLASS_NAMES)
        model = build_model(num_classes=len(classes), freeze_backbone=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        print(f"Loaded model from {MODEL_PATH} | classes={classes}")
    else:
        print(f"WARNING: no trained model found at {MODEL_PATH}. "
              f"/predict will return an error until a model is trained and placed there.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "classes": classes}


@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train and place best_model.pt first.")

    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    # Grad-CAM on last conv block of ResNet18
    cam_engine = GradCAM(model, target_layer=model.layer4[-1])
    cam, class_idx, confidence = cam_engine.generate(input_tensor)

    predicted_class = classes[class_idx]
    focus_region = estimate_region(cam)

    # Build overlay image
    np_img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    overlay = overlay_heatmap(bgr_img, cam)

    gradcam_filename = f"{uuid.uuid4().hex}.png"
    gradcam_path = os.path.join(GRADCAM_DIR, gradcam_filename)
    cv2.imwrite(gradcam_path, overlay)

    # LLM-generated explanation report
    report_text = generate_report(predicted_class, confidence, focus_region)

    record = Prediction(
        image_filename=file.filename,
        predicted_class=predicted_class,
        confidence=confidence,
        focus_region=focus_region,
        gradcam_path=gradcam_filename,
        llm_report=report_text,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "focus_region": focus_region,
        "gradcam_image_url": f"/gradcam/{gradcam_filename}",
        "llm_report": report_text,
        "disclaimer": "This is an AI-assisted screening tool for educational/demo purposes only. "
                      "It is NOT a medical diagnosis. Consult a licensed physician.",
    }


@app.get("/history")
def get_history(limit: int = 20, db: Session = Depends(get_db)):
    records = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(limit).all()
    return [r.to_dict() for r in records]


@app.get("/history/{record_id}")
def get_history_item(record_id: int, db: Session = Depends(get_db)):
    record = db.query(Prediction).filter(Prediction.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record.to_dict()


@app.get("/gradcam/{filename}")
def get_gradcam_image(filename: str):
    path = os.path.join(GRADCAM_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)
