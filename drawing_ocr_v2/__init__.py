"""Drawing-OCR standalone package (v2): TrOCR + projection / CRAFT_Lite.

Domain-specific OCR for engineering drawing dimension and tolerance text.

Architecture:
  * Detector: horizontal projection splitter (default) OR our 3.7M-param
    CRAFT_Lite variant OR `auto` (both)
  * Recogniser: fine-tuned TrOCR (94.6 % full-match on our 74-image
    benchmark) — falls back to stock `microsoft/trocr-small-printed` if
    the local fine-tuned checkpoint isn't present
  * Rotation pass: re-runs detection + recognition on 90°/270° rotated
    frame to recover vertical-oriented text

Quick start:
    >>> from drawing_ocr_v2 import OCR
    >>> import numpy as np
    >>> from PIL import Image
    >>> ocr = OCR()
    >>> img = np.asarray(Image.open("dimension.png").convert("RGB"))
    >>> for text, conf, bbox in ocr.recognize(img):
    ...     print(f"{conf:.2f}  {text}")
    0.99  Ø50
    0.97  ±0.05
"""

__version__ = "2.0.0"

from .api import OCR

__all__ = ["OCR"]
