import cv2
import numpy as np
import torch
from ultralytics import YOLO
import segmentation_models_pytorch as smp

# ---- Config ----
CLASS_MAP = {0: 'Animal', 1: 'Waste', 2: 'Person'}
NUM_CLASSES = len(CLASS_MAP) + 1  # +1 background

# BGR colors (OpenCV format) - Animal=red, Waste=green, Person=blue
CLASS_COLORS = {
    0: (0, 0, 255),    # Animal - red
    1: (0, 255, 0),    # Waste - green
    2: (255, 0, 0),    # Person - blue
}

IMG_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

YOLO_WEIGHTS_PATH = 'models/best_yolo.pt'
UNET_WEIGHTS_PATH = 'models/best_unet.pt'

# ---- Load models once at startup ----
_yolo_model = None
_unet_model = None


def load_models():
    global _yolo_model, _unet_model
    _yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    _unet_model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,  # weights will be loaded from checkpoint
        in_channels=3,
        classes=NUM_CLASSES
    )
    _unet_model.load_state_dict(torch.load(UNET_WEIGHTS_PATH, map_location=DEVICE))
    _unet_model.to(DEVICE)
    _unet_model.eval()

    print("Models loaded.")


def _normalize_img(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_rgb = (img_rgb - mean) / std
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def _make_overlay(img_bgr, class_mask, alpha=0.5):
    """class_mask: HxW array where 0=background, 1..N = class ids (matching CLASS_MAP+1)"""
    overlay = img_bgr.copy()
    color_layer = np.zeros_like(img_bgr)

    for cls_id, color in CLASS_COLORS.items():
        color_layer[class_mask == (cls_id + 1)] = color

    mask_present = class_mask > 0
    overlay[mask_present] = cv2.addWeighted(img_bgr, 1 - alpha, color_layer, alpha, 0)[mask_present]
    return overlay


def _compute_stats(class_mask):
    total_pixels = class_mask.size
    stats = {}
    for cls_id, name in CLASS_MAP.items():
        pixel_count = int((class_mask == (cls_id + 1)).sum())
        stats[name] = {
            "pixel_count": pixel_count,
            "pixel_percent": round(100 * pixel_count / total_pixels, 2)
        }
    return stats


def run_yolo_inference(img_bgr):
    h, w = img_bgr.shape[:2]
    results = _yolo_model.predict(source=img_bgr, verbose=False)
    r = results[0]

    class_mask = np.zeros((h, w), dtype=np.uint8)
    instance_count = {name: 0 for name in CLASS_MAP.values()}

    if r.masks is not None:
        for mask, cls in zip(r.masks.data, r.boxes.cls):
            cls_id = int(cls)
            m = cv2.resize(mask.cpu().numpy(), (w, h))
            class_mask[m > 0.5] = cls_id + 1
            instance_count[CLASS_MAP[cls_id]] += 1

    overlay = _make_overlay(img_bgr, class_mask)
    stats = _compute_stats(class_mask)
    for name in stats:
        stats[name]["instance_count"] = instance_count[name]

    return overlay, stats


def run_unet_inference(img_bgr):
    h, w = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    tensor = _normalize_img(resized).to(DEVICE)

    with torch.no_grad():
        output = _unet_model(tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    class_mask = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = _make_overlay(img_bgr, class_mask)
    stats = _compute_stats(class_mask)

    return overlay, stats
