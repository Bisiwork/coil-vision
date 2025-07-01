# src/vision/preprocessing.py

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

def preprocess_variant(
    img: np.ndarray,
    *,
    upscale: float,
    clipLimit: float,
    tileGridSize: tuple[int,int],
    threshold: str,
    blockSize: int,
    C: int,
    blur: str|None,
    erosion_kernel: tuple[int,int]
) -> np.ndarray:
    # 1) upscale
    img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    # 2) grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3) optional blur
    if blur == "gaussian":
        gray = cv2.GaussianBlur(gray, (5,5), 0)
    elif blur == "median":
        gray = cv2.medianBlur(gray, 5)
    # 4) CLAHE
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    contrast = clahe.apply(gray)
    # 5) threshold
    if threshold == "otsu":
        _, bin_img = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold == "adaptive_mean":
        bin_img = cv2.adaptiveThreshold(
            contrast, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, blockSize, C
        )
    else:  # adaptive_gaussian
        bin_img = cv2.adaptiveThreshold(
            contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize, C
        )
    # 6) erosion per assottigliare
    kernel = np.ones(erosion_kernel, np.uint8)
    thinned = cv2.erode(bin_img, kernel, iterations=1)
    return thinned

def generate_preprocessing_variants(
    image_path: str,
    output_dir: str = "src/input/preprocessed_variants"
) -> List[Path]:
    """
    Genera 10 preprocessamenti diversi e li salva in output_dir.
    Ritorna la lista dei Path generati.
    """
    configs: List[Dict[str, Any]] = [
        {"upscale":2.0, "clipLimit":1.0, "tileGridSize":(8,8), "threshold":"otsu",           "blockSize":11, "C":2, "blur":None,         "erosion_kernel":(1,1)},
        {"upscale":2.0, "clipLimit":2.0, "tileGridSize":(8,8), "threshold":"otsu",           "blockSize":11, "C":2, "blur":"gaussian",   "erosion_kernel":(1,1)},
        {"upscale":3.0, "clipLimit":2.0, "tileGridSize":(8,8), "threshold":"adaptive_mean",  "blockSize":15, "C":5, "blur":None,         "erosion_kernel":(2,2)},
        {"upscale":3.0, "clipLimit":2.0, "tileGridSize":(8,8), "threshold":"adaptive_gaussian","blockSize":15, "C":5, "blur":None,         "erosion_kernel":(2,2)},
        {"upscale":3.0, "clipLimit":3.0, "tileGridSize":(4,4), "threshold":"otsu",           "blockSize":11, "C":2, "blur":"median",     "erosion_kernel":(1,1)},
        {"upscale":2.5, "clipLimit":1.5, "tileGridSize":(8,8), "threshold":"adaptive_mean",  "blockSize":21, "C":3, "blur":"gaussian",   "erosion_kernel":(1,1)},
        {"upscale":2.5, "clipLimit":1.5, "tileGridSize":(8,8), "threshold":"adaptive_gaussian","blockSize":21, "C":3, "blur":"median",     "erosion_kernel":(1,1)},
        {"upscale":3.0, "clipLimit":2.0, "tileGridSize":(16,16),"threshold":"otsu",          "blockSize":11, "C":0, "blur":None,        "erosion_kernel":(2,2)},
        {"upscale":2.0, "clipLimit":2.5, "tileGridSize":(8,8), "threshold":"adaptive_mean",  "blockSize":11, "C":5, "blur":"gaussian",   "erosion_kernel":(1,1)},
        {"upscale":2.0, "clipLimit":2.5, "tileGridSize":(8,8), "threshold":"adaptive_gaussian","blockSize":11, "C":5, "blur":"median",     "erosion_kernel":(1,1)},
    ]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    img = cv2.imread(image_path)
    out_paths: List[Path] = []

    for idx, cfg in enumerate(configs, start=1):
        proc = preprocess_variant(img.copy(), **cfg)
        out_path = Path(output_dir) / f"variant_{idx}.png"
        cv2.imwrite(str(out_path), proc)
        out_paths.append(out_path)

    return out_paths
