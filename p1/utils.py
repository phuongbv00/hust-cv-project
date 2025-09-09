import shutil
from pathlib import Path

import cv2
import numpy as np


def read_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def ensure_output_dir(input_path: str, recreate=True) -> Path:
    project_root = Path(__file__).resolve().parent
    out_root = project_root / "output"
    try:
        out_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Failed to create output root {out_root}: {e}")
    stem = Path(input_path).name
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    out_dir = out_root / stem
    try:
        if recreate and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Failed to create output dir {out_dir}: {e}")
    return out_dir


def save_image(path: Path, img: np.ndarray):
    try:
        if img.dtype != np.uint8:
            imin, imax = float(np.min(img)), float(np.max(img))
            if imax > imin:
                norm = (img - imin) / (imax - imin)
            else:
                norm = np.zeros_like(img, dtype=np.float32)
            img_to_save = (norm * 255).astype(np.uint8)
        else:
            img_to_save = img
        cv2.imwrite(str(path), img_to_save)
    except Exception as e:
        print(f"Failed to save {path}: {e}")
