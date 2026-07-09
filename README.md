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


