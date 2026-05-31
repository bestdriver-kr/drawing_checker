"""Drawing OCR — standalone inference engine.

High-level API
==============
    from drawing_ocr import DrawingOCR

    ocr = DrawingOCR()                       # auto: load model from models/
    text, conf = ocr.recognize_line(crop)    # one text-line crop → string
    items = ocr.recognize_image(image)       # full image → list of fragments

`recognize_line` takes a SINGLE line of text already cropped (any size —
internally normalised to 32-px height) and returns one (text, confidence)
pair.  Use it when your application has already located the text region.

`recognize_image` takes a FULL image (any size) and returns multiple
(text, conf, bbox) fragments.  It needs a text detector to find the
bboxes.  By default it uses EasyOCR's CRAFT detector if `easyocr` is
installed; otherwise it falls back to projection-based whole-image line
splitting (works for clean single/multi-line crops).

The recogniser model is the CRNN trained for engineering-drawing digits
+ tolerance symbols + ISO 286 letters.  Charset is 72 classes — see
`drawing_ocr.charset.CHARSET` for the full list.
"""

from __future__ import annotations

import math
import os
import threading
from typing import List, Tuple

import numpy as np
import torch

from .charset import BLANK_IDX, _IDX_TO_CHAR  # type: ignore
from .model import build_model
from .line_detector import split_box_into_lines, detect_lines_on_whole
from . import preprocess


# ---------------------------------------------------------------------------
# CRNN input normalisation
# ---------------------------------------------------------------------------
TARGET_H = 32          # all line crops resize to this pixel height
_PAD_X = 4             # blank padding on each side so CTC has framing room


def _prep_crop(crop_rgb: np.ndarray) -> torch.Tensor:
    """Convert a colour crop to the (1, 1, 32, W) float tensor the CRNN expects.

    Background-aware inversion is applied (dark CAD → invert) so the
    recogniser always sees black-on-white text regardless of the source.
    """
    try:
        import cv2
        if crop_rgb.ndim == 3:
            gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop_rgb
    except ImportError:
        # No OpenCV — use PIL fallback
        from PIL import Image
        if crop_rgb.ndim == 3:
            gray = np.asarray(Image.fromarray(crop_rgb).convert("L"))
        else:
            gray = crop_rgb

    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return torch.zeros(1, 1, TARGET_H, TARGET_H, dtype=torch.float32)
    if gray.mean() < 110:
        gray = 255 - gray
    new_w = max(8, int(round(w * (TARGET_H / h))))
    try:
        import cv2
        gray = cv2.resize(gray, (new_w, TARGET_H), interpolation=cv2.INTER_LINEAR)
        if _PAD_X > 0:
            gray = cv2.copyMakeBorder(gray, 0, 0, _PAD_X, _PAD_X,
                                      cv2.BORDER_CONSTANT, value=255)
    except ImportError:
        from PIL import Image
        gray = np.asarray(Image.fromarray(gray).resize((new_w, TARGET_H)))
        gray = np.pad(gray, ((0, 0), (_PAD_X, _PAD_X)),
                      mode="constant", constant_values=255)
    arr = gray.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# CTC greedy decode with per-result confidence
# ---------------------------------------------------------------------------
def _decode_with_conf(log_probs: torch.Tensor) -> Tuple[str, float]:
    """log_probs: (T, C) tensor → (text, mean-exp-logprob confidence)."""
    idx = log_probs.argmax(dim=-1)
    best_lp = log_probs.gather(1, idx.unsqueeze(1)).squeeze(1)
    chars: List[str] = []
    char_lp: List[float] = []
    prev = -1
    for t in range(idx.size(0)):
        i = int(idx[t].item())
        if i != prev and i != BLANK_IDX:
            chars.append(_IDX_TO_CHAR[i])
            char_lp.append(float(best_lp[t].item()))
        prev = i
    text = "".join(chars)
    if char_lp:
        conf = math.exp(sum(char_lp) / len(char_lp))
    else:
        conf = 0.0
    return text, max(0.0, min(1.0, conf))


# ---------------------------------------------------------------------------
# Main public class
# ---------------------------------------------------------------------------
def _default_model_path() -> str:
    """Locate the bundled checkpoint at `<package>/../models/drawing_crnn.pt`.

    Works whether the package is imported as a sibling of `models/` or
    installed under site-packages with the model copied alongside.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # Try two locations: ../models/ (development layout) and ./models/
    # (package-bundled layout).
    for candidate in (
        os.path.join(here, "..", "models", "drawing_crnn.pt"),
        os.path.join(here, "models", "drawing_crnn.pt"),
    ):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    # Nothing found — return the first candidate so the error message
    # points the user at the conventional location.
    return os.path.abspath(os.path.join(here, "..", "models", "drawing_crnn.pt"))


class DrawingOCR:
    """Thread-safe, lazy-loading OCR engine for engineering drawings.

    Parameters
    ----------
    model_path : str | None
        Path to the CRNN checkpoint.  Default: look beside the package
        in a `models/` directory.
    device : str
        'cuda' | 'cpu' | 'auto'.  'auto' picks CUDA when available.
    easy_reader : easyocr.Reader | None
        Optional pre-built EasyOCR Reader for text detection.  If None,
        the engine tries to import + build one lazily on first
        recognize_image call.  Pass `False` to disable detection entirely
        and rely on projection-based whole-image line splitting.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
        easy_reader=None,
    ):
        self._model_path = model_path or _default_model_path()
        self._device = self._resolve_device(device)
        self._model = None
        self._model_lock = threading.Lock()
        self._easy_reader = easy_reader  # None = lazy; False = disabled

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name == "cuda":
            return torch.device("cuda")
        if name == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Lazy load ────────────────────────────────────────────────────────
    def _get_model(self):
        with self._model_lock:
            if self._model is not None:
                return self._model
            if not os.path.isfile(self._model_path):
                raise FileNotFoundError(
                    f"Drawing OCR checkpoint not found: {self._model_path}\n"
                    "Copy the .pt file to the models/ folder next to this "
                    "package, or pass model_path=... explicitly."
                )
            ckpt = torch.load(self._model_path, map_location=self._device,
                              weights_only=False)
            model = build_model(self._device)
            model.load_state_dict(ckpt.get("state_dict", ckpt))
            model.eval()
            self._model = model
            return model

    def _get_easy_reader(self):
        if self._easy_reader is False:
            return None
        if self._easy_reader is not None:
            return self._easy_reader
        try:
            import easyocr
            use_gpu = self._device.type == "cuda"
            self._easy_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
            return self._easy_reader
        except Exception as e:
            print(f"[drawing_ocr] EasyOCR detector unavailable: {e}")
            print("[drawing_ocr] Falling back to projection-based line splitting.")
            self._easy_reader = False
            return None

    # ── Single-line recognition ──────────────────────────────────────────
    def recognize_line(self, crop_rgb: np.ndarray) -> Tuple[str, float]:
        """Recognise one text line crop.  Returns (text, confidence ∈ [0,1])."""
        model = self._get_model()
        t = _prep_crop(crop_rgb).to(self._device, non_blocking=True)
        with torch.no_grad():
            logits = model(t)
        return _decode_with_conf(logits[:, 0, :].cpu())

    def recognize_lines(
        self, crops: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        """Batched version of `recognize_line`.  More efficient on GPU when
        you have several crops to read at once."""
        if not crops:
            return []
        model = self._get_model()
        tensors = [_prep_crop(c) for c in crops]
        max_w = max(t.shape[-1] for t in tensors)
        batch = torch.ones(len(tensors), 1, TARGET_H, max_w, dtype=torch.float32)
        for i, t in enumerate(tensors):
            batch[i, :, :, :t.shape[-1]] = t
        batch = batch.to(self._device, non_blocking=True)
        with torch.no_grad():
            logits = model(batch)
        return [_decode_with_conf(logits[:, i, :].cpu())
                for i in range(len(crops))]

    # ── Full-image recognition ───────────────────────────────────────────
    def recognize_image(
        self,
        image_rgb: np.ndarray,
        *,
        use_variants: bool = True,
        bg_mode: str = "auto",
    ) -> List[Tuple[str, float, List[List[int]], str]]:
        """Recognise text in a full image.

        Returns a list of (text, confidence, bbox-4pt-polygon, source-tag)
        tuples, sorted by confidence descending.  When `use_variants=True`
        (default), the image is preprocessed in several ways (upscale,
        binarise, dilate, …) and the recogniser runs on each — pooled
        results catch glyphs that one variant misses.  Set False for a
        single fast pass on the raw image.
        """
        easy_reader = self._get_easy_reader()
        variants = (preprocess.build_variants(image_rgb, bg_mode=bg_mode)
                    if use_variants else [("raw", image_rgb, 1.0)])
        pooled = []
        seen: set = set()

        for tag, vimg, scale in variants:
            raw_boxes, src = self._detect(vimg, easy_reader)
            crops, bboxes = [], []
            allow_split = (src == "craft")
            for rb in raw_boxes:
                for crop, bb in self._split_and_crop(vimg, rb,
                                                     allow_split=allow_split):
                    crops.append(crop)
                    bboxes.append(bb)
            if not crops:
                continue
            results = self.recognize_lines(crops)
            for (text, conf), bbox in zip(results, bboxes):
                text = text.strip()
                if not text:
                    continue
                key = (text, round(conf, 2))
                if key in seen:
                    continue
                seen.add(key)
                pooled.append((text, conf, bbox, f"{src}-{tag}"))

        pooled.sort(key=lambda x: -x[1])
        return pooled

    # ── Detection + crop helpers ─────────────────────────────────────────
    @staticmethod
    def _detect(vimg, easy_reader):
        """Run EasyOCR detector when available, fall back to projection
        splitting otherwise.  Returns (axis-aligned boxes, source-tag)."""
        if easy_reader is not None:
            try:
                horizontal_list, _free_list = easy_reader.detect(
                    vimg,
                    min_size=15, text_threshold=0.65, low_text=0.4,
                    link_threshold=0.4, mag_ratio=2.0,
                    slope_ths=0.1, ycenter_ths=0.3,
                    height_ths=0.5, width_ths=0.5, add_margin=0.08,
                )
            except TypeError:
                # Older EasyOCR may reject some kwargs
                horizontal_list, _free_list = easy_reader.detect(vimg)
            except Exception:
                horizontal_list = None
            if horizontal_list and horizontal_list[0]:
                raw = []
                for box in horizontal_list[0]:
                    x_min, x_max, y_min, y_max = [int(v) for v in box]
                    raw.append((x_min, y_min, x_max, y_max))
                if raw:
                    return raw, "craft"
        # Fallback: projection-based whole-image line splitting
        return detect_lines_on_whole(vimg), "projection"

    @staticmethod
    def _split_and_crop(vimg, raw_box_xyxy, *, allow_split: bool):
        x_min, y_min, x_max, y_max = raw_box_xyxy
        x_min = max(0, int(x_min)); y_min = max(0, int(y_min))
        x_max = min(vimg.shape[1], int(x_max))
        y_max = min(vimg.shape[0], int(y_max))
        if x_max <= x_min or y_max <= y_min:
            return []
        if allow_split:
            sub_boxes = split_box_into_lines(vimg, x_min, y_min, x_max, y_max)
        else:
            sub_boxes = [(x_min, y_min, x_max, y_max)]
        out = []
        for sx0, sy0, sx1, sy1 in sub_boxes:
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            crop = vimg[sy0:sy1, sx0:sx1]
            bbox = [[sx0, sy0], [sx1, sy0], [sx1, sy1], [sx0, sy1]]
            out.append((crop, bbox))
        return out
