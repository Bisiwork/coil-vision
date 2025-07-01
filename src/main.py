# src/main.py

import os
from pathlib import Path
from dotenv import load_dotenv
from vision.preprocessing import preprocess_image
from vision.ocr import extract_text
from tablegen.gpt_table_parser import generate_table_from_ocr_text
from tablegen.table_builder import save_table_as_csv

load_dotenv()

OCR_METHODS = [
    "tesseract",
    "easyocr",
    "paddleocr",
    "opencv_tesseract",
    "openai_vision",
    ]

file_name = "sample1.png"
input_dir = Path("src/input")
output_dir = Path("src/output")

INPUT_IMAGE = input_dir / file_name
basename = file_name.rsplit(".", 1)[0]

def main():
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(f"Immagine non trovata: {INPUT_IMAGE}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for method in OCR_METHODS:
        print(f"\n=== Metodo OCR: {method} ===")
        try:
            preprocessed_image = preprocess_image(str(INPUT_IMAGE))
            ocr_text = extract_text(preprocessed_image, method=method)
        except Exception as e:
            print(f"❌ Errore OCR con metodo '{method}': {e}")
            continue

        ocr_txt_path = output_dir / f"ocr_{method}_{file_name}.txt"
        with open(ocr_txt_path, "w", encoding="utf-8") as f:
            f.write(ocr_text)
        print(f"✓ OCR text saved to {ocr_txt_path}")

        try:
            rows = generate_table_from_ocr_text(ocr_text)
        except Exception as e:
            print(f"❌ Errore GPT parsing con metodo '{method}': {e}")
            continue

        if not rows:
            print(f"⚠️ Nessuna riga generata con metodo '{method}'")
            continue

        print(f"✓ {len(rows)} righe generate con metodo '{method}'")
        csv_path = output_dir / f"{method}_{basename}_table.csv"
        save_table_as_csv(rows, str(csv_path))

    print("\n✅ Tutti i metodi OCR testati. Controlla i file in src/output/")

if __name__ == "__main__":
    main()
