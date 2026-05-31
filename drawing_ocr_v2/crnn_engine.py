"""CRNN recognition wrapper — light, fast, fully self-contained.

Uses the in-house drawing-OCR CRNN (7.5 M params, ~30 MB).  No external
model downloads, no HuggingFace dependency.  Lower accuracy than TrOCR
(~73 % vs 94.6 % on our benchmark) but faster (~50 ms vs ~500 ms per
line on CUDA) and ~8× smaller.
"""

from __future__ import annotations

import math
import os
import threading
from typing import List, Tuple

import numpy as np
import torch

from .charset import BLANK_IDX, _IDX_TO_CHAR
from .model import build_model


TARGET_H = 32
_PAD_X = 4

_MODEL = None
_DEVICE = None
_LOCK = threading.Lock()


def _default_model_path() -> str:
    """models/drawing_crnn.pt next to this package."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "models", "drawing_crnn.pt")


def _get_crnn(model_path: str | None = None, prefer_cuda: bool = True):
    """Build (or return cached) CRNN model.  Thread-safe."""
    global _MODEL, _DEVICE
    with _LOCK:
        target = torch.device("cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu")
        if _MODEL is not None and _DEVICE == target:
            return _MODEL
        path = model_path or _default_model_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"CRNN checkpoint not found: {path}\n"
                "  Provide model_path= or copy drawing_crnn.pt into models/"
            )
        ckpt = torch.load(path, map_location=target, weights_only=False)
        model = build_model(target)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        model.eval()
        _MODEL = model
        _DEVICE = target
        return _MODEL


def _prep_crop(crop_rgb: np.ndarray) -> torch.Tensor:
    """RGB crop → (1, 1, 32, W) float tensor, background-aware inverted."""
    try:
        import cv2
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if crop_rgb.ndim == 3 else crop_rgb
    except ImportError:
        from PIL import Image
        gray = np.asarray(Image.fromarray(crop_rgb).convert("L")) if crop_rgb.ndim == 3 else crop_rgb

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
        gray = np.asarray(
            Image.fromarray(gray).resize((new_w, TARGET_H), Image.BILINEAR)
        )
        if _PAD_X > 0:
            pad = np.full((TARGET_H, new_w + 2 * _PAD_X), 255, dtype=gray.dtype)
            pad[:, _PAD_X:_PAD_X + new_w] = gray
            gray = pad

    arr = gray.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def _decode_with_conf(log_probs: torch.Tensor) -> Tuple[str, float]:
    """log_probs: (T, C) → (text, mean-exp-logprob confidence)."""
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
    conf = math.exp(sum(char_lp) / len(char_lp)) if char_lp else 0.0
    return text, max(0.0, min(1.0, conf))


def recognize_lines(
    crops_rgb: List[np.ndarray],
    *,
    model_path: str | None = None,
    prefer_cuda: bool = True,
) -> List[Tuple[str, float]]:
    """Batch-recognise a list of line crops with the CRNN."""
    if not crops_rgb:
        return []
    model = _get_crnn(model_path, prefer_cuda)
    device = _DEVICE
    tensors = [_prep_crop(c) for c in crops_rgb]
    max_w = max(t.shape[-1] for t in tensors)
    batch = torch.ones(len(tensors), 1, TARGET_H, max_w, dtype=torch.float32)
    for i, t in enumerate(tensors):
        batch[i, :, :, :t.shape[-1]] = t
    batch = batch.to(device, non_blocking=True)
    with torch.no_grad():
        log_probs = torch.log_softmax(model(batch), dim=-1).cpu()
    return [_decode_with_conf(log_probs[:, i, :]) for i in range(len(crops_rgb))]
