#!/usr/bin/env python3
"""
coil_vision.py   ·   CSV-ready pipeline

1.  PDF→PNG (400 DPI)
2.  Detect black text/geometry regions
3.  Compute and save distance map transform for debug
4.  Crop + OCR (Tesseract)
5.  Merge OCR testo → stringa
6.  Pass OCR string to gpt_table_parser.generate_table_from_ocr_text()
7.  Save CSV with table_builder.save_table_as_csv()

CSV header (fixed order):
    step,feed,feed_abs,diameter,rotation,
    v_pitch,h_pitch,mand_y,mand_z,funct,param,speed
"""
from __future__ import annotations
import base64
import json
import logging
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path

# ── custom table-generator tools ──────────────────────────────────────────────
from tablegen.gpt_table_parser import generate_table_from_ocr_text
from tablegen.table_builder    import save_table_as_csv

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path("/Users/stefanoroybisignano/Desktop/W_SimplexRapid/coil-vision/")
INPUT_DIR     = BASE_DIR / "data/raw"
CROPS_DIR     = BASE_DIR / "data/processed/crops"
OCR_DIR       = BASE_DIR / "data/ocr/json"
DEBUG_DIR     = BASE_DIR / "data/debug"

DPI, MIN_AREA, MIN_W, MIN_H = 400, 2_000, 40, 40
ASPECT_RANGE, KERNEL_SIZE = (0.25, 4.0), (50, 5)
TEXT_DENSITY_MIN, TESSERACT_CONF = 0.02, "--psm 6"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ── HELPERS ───────────────────────────────────────────────────────────────────
def ensure(*dirs: Path):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def pdf_to_images(pdf: Path) -> List[np.ndarray]:
    """Convert PDF to list of images (NumPy arrays)."""
    logging.info(f"[PDF→IMG] {pdf.name}")
    pages = convert_from_path(str(pdf), dpi=DPI)
    imgs = []
    for pg in pages:
        arr = np.array(pg)
        imgs.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return imgs


def detect_black_regions(img: np.ndarray) -> Tuple[List[Tuple[int,int,int,int]], np.ndarray, np.ndarray]:
    """Return boxes around black regions, plus binary mask and dilated mask."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE), iterations=2)

    stats = cv2.connectedComponentsWithStats(dil, connectivity=8)[2]
    boxes: List[Tuple[int,int,int,int]] = []
    for x, y, w, h, area in stats[1:]:
        x, y, w, h, area = map(int, (x, y, w, h, area))
        if area < MIN_AREA or w < MIN_W or h < MIN_H:
            continue
        ar = w / h
        if not (ASPECT_RANGE[0] <= ar <= ASPECT_RANGE[1]):
            continue
        boxes.append((x, y, x + w, y + h))

    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes, mask, dil


def compute_and_save_distance_map(sample: str, page: int, mask: np.ndarray):
    """
    Compute distance transform from black regions and save to DEBUG_DIR.
    """
    dbg_dir = DEBUG_DIR / sample
    dbg_dir.mkdir(parents=True, exist_ok=True)
    # background is 255-mask
    bg = cv2.bitwise_not(mask)
    dist = cv2.distanceTransform(bg, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # normalize to 0–255
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    path = dbg_dir / f"{sample}_p{page}_distmap.png"
    cv2.imwrite(str(path), dist_norm)
    logging.info(f"   Saved distance map → {path.name}")


def crop_and_ocr(img: np.ndarray, boxes: List[Tuple[int,int,int,int]], crop_dir: Path, prefix: str) -> List[str]:
    """Return list of OCR strings (one per kept crop)."""
    texts: List[str] = []
    crop_dir.mkdir(parents=True, exist_ok=True)
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        if inv.mean() / 255.0 < TEXT_DENSITY_MIN:
            continue
        fn = f"{prefix}_{i:02d}.png"
        cv2.imwrite(str(crop_dir / fn), crop)
        texts.append(pytesseract.image_to_string(crop, config=TESSERACT_CONF).strip())
    return texts


# ── DRIVER ────────────────────────────────────────────────────────────────────
def process_file(src: Path):
    sample = src.stem
    logging.info(f"→ {sample}")
    ensure(CROPS_DIR, OCR_DIR, DEBUG_DIR)

    # 1) load images
    if src.suffix.lower() == ".pdf":
        images = pdf_to_images(src)
    else:
        images = [cv2.imread(str(src))]

    # 2) OCR text collection
    all_texts: List[str] = []
    for idx, img in enumerate(images, start=1):
        boxes, mask, dil = detect_black_regions(img)
        # save debug artefacts
        dbg_dir = DEBUG_DIR / sample
        ensure(dbg_dir)
        cv2.imwrite(str(dbg_dir / f"{sample}_p{idx}_mask.png"), mask)
        cv2.imwrite(str(dbg_dir / f"{sample}_p{idx}_dil.png"),  dil)
        # distance map
        compute_and_save_distance_map(sample, idx, mask)

        # crop+OCR
        crops_subdir = CROPS_DIR / sample
        texts = crop_and_ocr(img, boxes, crops_subdir, f"{sample}_p{idx}")
        all_texts.extend(texts)

    ocr_text = "\n".join(all_texts)

    # 3) generate table rows
    rows = generate_table_from_ocr_text(ocr_text)
    if not rows:
        logging.error(f"⚠️ GPT failed to build table for {sample}")
        return

    # 4) save CSV
    csv_path = OCR_DIR / f"{sample}_final.csv"
    save_table_as_csv(rows, str(csv_path))
    logging.info(f"   Saved final CSV → {csv_path.name}")


# ── ENTRY ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure(INPUT_DIR, CROPS_DIR, OCR_DIR, DEBUG_DIR)
    files = [p for ext in ("*.pdf","*.png","*.jpg","*.jpeg") for p in INPUT_DIR.glob(ext)]
    if not files:
        logging.error("No input files found in INPUT_DIR")
        exit(1)
    for f in files:
        process_file(f)
    logging.info("✓ All drawings processed.")
