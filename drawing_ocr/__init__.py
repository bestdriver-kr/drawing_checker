"""Drawing OCR — domain-specific OCR for engineering drawing dimensions.

Quick start
-----------
    from drawing_ocr import DrawingOCR

    ocr = DrawingOCR()
    items = ocr.recognize_image(my_image_as_rgb_numpy)
    for text, conf, bbox, tag in items:
        print(f"{conf:.2f}  {text}")

    # Or if you already have a single-line text crop:
    text, conf = ocr.recognize_line(line_crop)

What this engine recognises
---------------------------
72 character classes: digits 0-9, signs +/-/±, separators . , /, the
diameter symbol Ø, R for radius, °, x for multipliers, space, and the
full Latin alphabet a-z + A-Z (for ISO 286 fit codes like H7, g6, f7).

Trained on synthetic data (PIL-rendered Windows fonts) plus user
fine-tune samples.  Best-in-class on:
  - decimal dimensions (50.025, Ø3.28, 192.011)
  - tolerance notation (±0.05, +0.1/-0.05)
  - ISO 286 fit codes (H7, g6, f7) + explicit tolerances
  - small-diameter callouts (Ø1-9 with 2-3 decimal places)
"""

__version__ = "1.0.0"

from .engine import DrawingOCR
from .charset import CHARSET, NUM_CLASSES, BLANK_IDX, encode, decode

__all__ = [
    "DrawingOCR",
    "CHARSET", "NUM_CLASSES", "BLANK_IDX",
    "encode", "decode",
]
