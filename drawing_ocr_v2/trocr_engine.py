"""TrOCR recognition wrapper.

Loads a HuggingFace VisionEncoderDecoderModel (TrOCR family).  Prefers
a local fine-tuned checkpoint at `models/trocr_finetuned/` next to the
package, falling back to the stock `microsoft/trocr-small-printed` from
the HuggingFace Hub.

The fine-tuned variant is trained on engineering-drawing dimension text
and is typically 5-10 %p more accurate on that domain than the stock
model.  See `drawing_ocr.trocr_finetune` in the main project for the
reproducible training pipeline.
"""

from __future__ import annotations

import os
import threading
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image


_MODEL = None
_PROCESSOR = None
_DEVICE = None
_LOCK = threading.Lock()


def _default_model_path() -> str:
    """models/trocr_finetuned/ next to this package."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "models", "trocr_finetuned")


def _resolve_model_id(model_path: str | None) -> str:
    """Pick local fine-tuned dir if present, else fall back to HF Hub."""
    if model_path:
        return model_path
    local = _default_model_path()
    if os.path.isdir(local) and os.path.isfile(os.path.join(local, "config.json")):
        return local
    return "microsoft/trocr-small-printed"


def _get_trocr(model_path: str | None = None, prefer_cuda: bool = True):
    """Build (or return cached) TrOCR processor + model.  Thread-safe."""
    global _MODEL, _PROCESSOR, _DEVICE
    with _LOCK:
        target = torch.device("cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu")
        if _MODEL is not None and _DEVICE == target:
            return _PROCESSOR, _MODEL
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        model_id = _resolve_model_id(model_path)
        _PROCESSOR = TrOCRProcessor.from_pretrained(model_id)
        _MODEL = VisionEncoderDecoderModel.from_pretrained(model_id).to(target)
        _MODEL.eval()
        _DEVICE = target
        return _PROCESSOR, _MODEL


def recognize_lines(
    crops_rgb: List[np.ndarray],
    *,
    model_path: str | None = None,
    prefer_cuda: bool = True,
    max_length: int = 64,
) -> List[Tuple[str, float]]:
    """Recognise a list of single-line RGB crops with TrOCR.

    Returns list of (text, confidence) parallel to `crops_rgb`.
    Confidence is the geometric mean of token-level max probabilities.
    """
    if not crops_rgb:
        return []
    processor, model = _get_trocr(model_path, prefer_cuda)
    pils = [Image.fromarray(c) if isinstance(c, np.ndarray) else c for c in crops_rgb]
    pixel_values = processor(images=pils, return_tensors="pt").pixel_values.to(model.device)
    with torch.no_grad():
        out = model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
    texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
    # Confidence: geometric mean of per-step max token prob.
    if out.scores:
        import torch.nn.functional as F
        per_step_max = []
        for step_logits in out.scores:
            probs = F.softmax(step_logits, dim=-1)
            per_step_max.append(probs.max(dim=-1).values.log())  # (batch,)
        avg_logprob = torch.stack(per_step_max).mean(dim=0)
        confs = avg_logprob.exp().cpu().tolist()
    else:
        confs = [1.0] * len(texts)
    return list(zip([t.strip() for t in texts], confs))
