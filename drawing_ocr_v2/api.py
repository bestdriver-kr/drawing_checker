"""Unified OCR API for the drawing-OCR standalone package.

Combines:
  * detection — your choice of `projection` (default, best for drawings),
    `craft_lite` (our 3.7M-param CRAFT variant), or `auto` (both, deduped)
  * recognition — fine-tuned TrOCR if `models/trocr_finetuned/` is
    present, otherwise stock `microsoft/trocr-small-printed`
  * rotation pass — re-runs detection + recognition on the 90° / 270°
    rotated frame to recover vertical-oriented text (limit dimensions,
    callout text on diameter views).  This is the secret sauce that
    pushes our benchmark from 83.8 % → 94.6 % full-match on real
    engineering captures.

Quick start:
    from drawing_ocr_v2 import OCR
    ocr = OCR()
    results = ocr.recognize(image_rgb)
    for text, conf, bbox in results:
        print(f"{conf:.2f}  {text}")
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Literal

import numpy as np


_DETECTOR_MODES = ("projection", "craft_lite", "auto")
_RECOGNIZER_MODES = ("trocr", "crnn")


class OCR:
    """High-level OCR engine for engineering drawing text.

    Parameters
    ----------
    detector : 'projection' | 'craft_lite' | 'auto', default 'projection'
        Which text-line detector to use.  `projection` is the horizontal
        projection splitter (best end-to-end accuracy on our benchmark);
        `craft_lite` is our 3.7M-param CRAFT variant; `auto` runs both
        and deduplicates.
    recognizer : 'trocr' | 'crnn', default 'trocr'
        Which recogniser to use.  `trocr` is the fine-tuned TrOCR
        (94.6 % full-match, ~500 ms/line, 250 MB model).  `crnn` is the
        in-house CRNN (73 % full-match, ~50 ms/line, 30 MB model) —
        choose this when speed/size matters more than accuracy.
    trocr_model_path : str, optional
        Path to TrOCR checkpoint dir; defaults to `models/trocr_finetuned/`.
        Only used when `recognizer='trocr'`.
    crnn_model_path : str, optional
        Path to `drawing_crnn.pt`; defaults to `models/drawing_crnn.pt`.
        Only used when `recognizer='crnn'`.
    craft_lite_path : str, optional
        Path to `craft_lite.pt`; defaults to `models/craft_lite.pt`.
        Only used when `detector='craft_lite'` or `'auto'`.
    device : 'auto' | 'cuda' | 'cpu', default 'auto'
    rotation_pass : bool, default True
        Whether to also try the 90°/270° rotated frame to recover
        vertical-oriented text.
    """

    def __init__(self,
                 detector: Literal["projection", "craft_lite", "auto"] = "projection",
                 recognizer: Literal["trocr", "crnn"] = "trocr",
                 trocr_model_path: Optional[str] = None,
                 crnn_model_path: Optional[str] = None,
                 craft_lite_path: Optional[str] = None,
                 device: Literal["auto", "cuda", "cpu"] = "auto",
                 rotation_pass: bool = True):
        if detector not in _DETECTOR_MODES:
            raise ValueError(
                f"detector must be one of {_DETECTOR_MODES}, got {detector!r}"
            )
        if recognizer not in _RECOGNIZER_MODES:
            raise ValueError(
                f"recognizer must be one of {_RECOGNIZER_MODES}, got {recognizer!r}"
            )
        self.detector = detector
        self.recognizer = recognizer
        self.trocr_model_path = trocr_model_path
        self.crnn_model_path = crnn_model_path
        self.craft_lite_path = craft_lite_path
        self.prefer_cuda = (device != "cpu")
        self.rotation_pass = rotation_pass
        self._craft_det = None  # lazy

    # ──────────────────────────────────────────────────────────────────
    # Detection
    # ──────────────────────────────────────────────────────────────────
    def _get_craft(self):
        if self._craft_det is None:
            from .craft_inference import CraftLiteDetector
            self._craft_det = CraftLiteDetector(
                checkpoint_path=self.craft_lite_path,
                device=("cuda" if self.prefer_cuda else "cpu"),
            )
        return self._craft_det

    def _detect(self, image_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        from .line_detector import detect_lines_on_whole
        if self.detector == "craft_lite":
            return self._get_craft().detect(image_rgb)
        if self.detector == "projection":
            return detect_lines_on_whole(image_rgb)
        # auto
        boxes = list(detect_lines_on_whole(image_rgb))
        if len(boxes) < 2:
            boxes.extend(self._get_craft().detect(image_rgb))
        return boxes

    # ──────────────────────────────────────────────────────────────────
    # Recognition
    # ──────────────────────────────────────────────────────────────────
    def _recognize_crops(self, crops_rgb):
        if self.recognizer == "trocr":
            from .trocr_engine import recognize_lines as rec_trocr
            return rec_trocr(
                crops_rgb,
                model_path=self.trocr_model_path,
                prefer_cuda=self.prefer_cuda,
            )
        from .crnn_engine import recognize_lines as rec_crnn
        return rec_crnn(
            crops_rgb,
            model_path=self.crnn_model_path,
            prefer_cuda=self.prefer_cuda,
        )

    # ──────────────────────────────────────────────────────────────────
    # Pipeline
    # ──────────────────────────────────────────────────────────────────
    def recognize(self,
                  image_rgb: np.ndarray,
                  ) -> List[Tuple[str, float, Optional[Tuple[int, int, int, int]]]]:
        """Recognise every text line in `image_rgb`.

        Returns a list of `(text, confidence, bbox)` tuples, sorted by
        confidence descending.  `bbox` is `(x_min, y_min, x_max, y_max)`
        in the original-image coordinate frame; `None` for results
        recovered via the rotation pass.
        """
        if image_rgb is None or image_rgb.size == 0:
            return []
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                "OCR.recognize expects a (H, W, 3) RGB numpy array"
            )

        results: List[Tuple[str, float, Optional[Tuple[int, int, int, int]]]] = []

        # Phase 1: native orientation
        boxes = self._detect(image_rgb)
        crops, bboxes = self._crop_boxes(image_rgb, boxes)
        if crops:
            for (text, conf), bbox in zip(self._recognize_crops(crops), bboxes):
                if text:
                    results.append((text, conf, bbox))

        # Phase 2: rotation pass (90° + 270°)
        if self.rotation_pass:
            results.extend(self._rotation_pass(image_rgb))

        # Dedup by normalised text, keep highest confidence
        return self._dedup_and_sort(results)

    def recognize_line(self,
                       crop_rgb: np.ndarray) -> Tuple[str, float]:
        """Recognise a single pre-cropped text line.  Returns (text, conf).

        Skips detection entirely — use when you already have the line
        bbox (e.g. from your own detector)."""
        res = self._recognize_crops([crop_rgb])
        return res[0] if res else ("", 0.0)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _crop_boxes(image_rgb, boxes):
        H, W = image_rgb.shape[:2]
        crops = []
        bboxes = []
        for box in boxes:
            x0, y0, x1, y1 = (int(v) for v in box)
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(W, x1); y1 = min(H, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crops.append(image_rgb[y0:y1, x0:x1])
            bboxes.append((x0, y0, x1, y1))
        return crops, bboxes

    def _rotation_pass(self, image_rgb):
        try:
            import cv2
        except ImportError:
            return []
        out = []
        for rot_deg, code in [
            (90, cv2.ROTATE_90_COUNTERCLOCKWISE),
            (270, cv2.ROTATE_90_CLOCKWISE),
        ]:
            try:
                rotated = cv2.rotate(image_rgb, code)
                boxes = self._detect(rotated)
                crops, _bb = self._crop_boxes(rotated, boxes)
                if not crops:
                    continue
                for text, conf in self._recognize_crops(crops):
                    if text:
                        # Bbox is in rotated frame, hard to map cleanly
                        # for arbitrary rotations — return None and let
                        # the caller use the text alone.
                        out.append((text, conf, None))
            except Exception as e:
                print(f"[ocr-v2] rotation pass {rot_deg}° failed: {e}")
        return out

    @staticmethod
    def _dedup_and_sort(results):
        sorted_ = sorted(results, key=lambda x: -x[1])
        seen = set()
        deduped = []
        for text, conf, bbox in sorted_:
            key = text.replace(" ", "").lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append((text, conf, bbox))
        return deduped
