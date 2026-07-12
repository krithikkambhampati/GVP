# Garbage & Waste Segmentation

Image segmentation project to detect and classify **Waste**, **Animal**, and **Person** in real-world scenes, aimed at supporting waste-monitoring and cleanup use cases.

## Status: In Progress

Currently comparing two segmentation architectures on the same dataset and evaluation split. Next phases will add SAM2/SAM3-based segmentation and a full-stack (MERN) web app for interactive predictions.

## Dataset

- ~1600 labeled images, 3 classes: `Waste`, `Animal`, `Person`
- Originally annotated as YOLO-format polygon labels (instance segmentation)
- Split: 70% train / 15% val / 15% test (fixed seed, reproducible)
- For UNet training, polygon labels were rasterized into pixel-wise class masks

*(Dataset not included in this repo due to size — stored in Google Drive.)*

## Models

### 1. YOLO11s-seg (Ultralytics)
- Instance segmentation, fine-tuned from COCO-pretrained weights
- Notebook: [`notebooks/Yolo.ipynb`](notebooks/Yolo.ipynb)

### 2. UNet (ResNet34 encoder)
- Semantic segmentation, ResNet34 encoder pretrained on ImageNet, fine-tuned
- Weighted loss (CrossEntropy + Dice) to address severe class imbalance (Animal = 0.07% of total pixels)
- Notebook: [`notebooks/Unet.ipynb`](notebooks/Unet.ipynb)

## Results

Since YOLO is instance-based and UNet is pixel-based, YOLO's predictions were collapsed into semantic masks for a fair pixel-wise IoU comparison.

| Class | YOLO11s-seg (IoU) | UNet-ResNet34 (IoU) |
|---|---|---|
| Background | – | 0.969 |
| Animal | 0.325 | 0.053 |
| Waste | 0.567 | 0.648 |
| Person | 0.440 | 0.335 |
| **Mean IoU (excl. background)** | **0.444** | **0.345** |

**Key finding:** YOLO outperforms UNet overall, driven almost entirely by the Animal class — UNet's plain pixel-wise learning struggled badly with Animal's extreme rarity in pixel coverage, while YOLO's instance-based detection handled it far better. UNet slightly outperforms YOLO on Waste, likely because Waste has larger, denser pixel regions well-suited to per-pixel classification.

Full metrics, confusion matrices, and training curves are in [`results/`](results/).

## Roadmap

- [x] YOLO11s-seg baseline
- [x] UNet (ResNet34) baseline
- [ ] SAM2 / SAM3-based segmentation
- [ ] MERN web app — upload an image, get live predictions from the trained models

## Setup

```bash
pip install -r requirements.txt
```

Notebooks are designed for Google Colab (GPU runtime) with dataset mounted from Google Drive.


# Garbage Segmentation Demo App

Upload an image, run it through your trained YOLO11s-seg or UNet (ResNet34) model, and view/download the segmentation overlay.

## Setup

### 1. Add your trained model weights

Copy your trained weights into the backend models folder:

```
backend/models/best_yolo.pt   <- from garbage_runs/seg_v1/weights/best.pt
backend/models/best_unet.pt   <- from garbage_runs/unet_v1/best_unet.pt
```

### 2. Backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend will run at `http://localhost:8000`. Check it's alive at `http://localhost:8000/` — should return `{"status": "ok"}`.

### 3. Frontend (React + Vite)

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend will run at `http://localhost:5173`. Open that in your browser.

## Usage

1. Upload an image
2. Select YOLO11s-seg or UNet (ResNet34)
3. Click Predict
4. View the overlay (red = Animal, green = Waste, blue = Person) and per-class pixel percentage
5. Click Download to save the result image

## Notes

- First request after starting the backend may be slower (model warm-up).
- If you get a CUDA error and don't have a GPU locally, the backend automatically falls back to CPU (inference will just be slower).
- CORS is configured for `localhost:5173` and `localhost:3000` only — update `main.py` if you serve the frontend elsewhere.
