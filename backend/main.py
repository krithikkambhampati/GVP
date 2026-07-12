import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference import load_models, run_yolo_inference, run_unet_inference

app = FastAPI(title="Garbage Segmentation API")

# Allow the React dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    load_models()


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form(...)):
    if model not in ("yolo", "unet"):
        raise HTTPException(status_code=400, detail="model must be 'yolo' or 'unet'")

    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    if model == "yolo":
        overlay, stats = run_yolo_inference(img_bgr)
    else:
        overlay, stats = run_unet_inference(img_bgr)

    success, buffer = cv2.imencode('.png', overlay)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode result image")

    overlay_b64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse({
        "model": model,
        "overlay_image": f"data:image/png;base64,{overlay_b64}",
        "stats": stats
    })
