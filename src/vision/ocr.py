# src/vision/ocr.py

from typing import Literal
from pathlib import Path
from PIL import Image
import pytesseract
import cv2
import numpy as np

# 1) Tesseract (locale, puro)
def run_tesseract(image_path: str) -> str:
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)

# 2) EasyOCR
def run_easyocr(image_path: str) -> str:
    import easyocr
    reader = easyocr.Reader(['en', 'it'])
    results = reader.readtext(image_path, detail=0)
    return "\n".join(results)

# 3) PaddleOCR
def run_paddleocr(image_path: str) -> str:
    from paddleocr import PaddleOCR

    # Inizializza la pipeline con classificazione dell’angolo
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    # Anche se è deprecato, questa chiamata funziona senza errori finché
    # non passeremo argomenti errati come 'cls' a predict()
    results = ocr.ocr(image_path)  

    # results: List[List[Tuple[bbox, (text, score)]]]
    extracted = []
    for line in results:
        for bbox, (text, score) in line:
            extracted.append(text)
    return "\n".join(extracted)




# 4) OpenAI Vision API (GPT-4o-mini)
def run_openai_vision(image_path: str) -> str:
    import os, base64
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Extract *exactly* the raw text from the image below, "
                    "preserving line breaks and numbers, with no commentary or formatting."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }],
        max_tokens=2000,
        response_format={"type": "text"}
    )
    return response.choices[0].message.content


# 5) Tesseract via OpenCV pre-processing
def run_opencv_tesseract(image_path: str) -> str:
    # Carica in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Denoise
    img = cv2.fastNlMeansDenoising(img, None, h=30)
    # Threshold adattivo
    img = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    # Opzionale: dilata/erode per migliorare contorni
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # Converti in PIL per Tesseract
    pil_img = Image.fromarray(img)
    return pytesseract.image_to_string(pil_img)


def run_docling_mode(image_path: str) -> str:
    """
    Esegue OCR + layout + struttura tabella.
    Per ora simula il comportamento con OCR + annotazione layout semplificata.
    """
    from vision.ocr import run_opencv_tesseract
    from docling.docling_layout import extract_layout
    from docling.docling_table_parser import build_docling_rows

    # Step 1: OCR (raw text)
    raw_text = run_opencv_tesseract(image_path)

    # Step 2: Layout analysis (es. zone, header, table)
    layout_info = extract_layout(image_path, raw_text)

    # Step 3: Riconoscimento tabellare (ritorna righe già strutturate)
    rows = build_docling_rows(raw_text, layout_info)
    return raw_text, rows  # restituisce entrambi




# Wrapper: scegliere modalità
def extract_text(image_path: str,
                 method: Literal[
                     "tesseract",
                     "easyocr",
                     "paddleocr",
                     "openai_vision",
                     "opencv_tesseract"
                 ] = "tesseract"
) -> str:
    if method == "tesseract":
        return run_tesseract(image_path)
    elif method == "easyocr":
        return run_easyocr(image_path)
    elif method == "paddleocr":
        return run_paddleocr(image_path)
    elif method == "openai_vision":
        return run_openai_vision(image_path)
    elif method == "opencv_tesseract":
        return run_opencv_tesseract(image_path)
    else:
        raise ValueError(f"OCR method '{method}' not recognized")
