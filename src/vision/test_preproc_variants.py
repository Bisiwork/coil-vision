# src/vision/test_preproc_variants.py

import sys
from pathlib import Path

# Assicuriamoci che `src/` sia nel path
root = Path(__file__).parents[1]  # sale da src/vision a src/
sys.path.insert(0, str(root))

from vision.preprocessing import generate_preprocessing_variants

if __name__ == "__main__":
    variants = generate_preprocessing_variants(str(root / "input" / "sample1.png"))
    print("Generated variants:")
    for p in variants:
        print(" -", p)
