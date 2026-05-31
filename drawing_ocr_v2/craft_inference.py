"""Inference + bounding-box extraction for the trained CRAFT_Lite detector.

Mirrors the API surface of EasyOCR's `reader.detect()` so we can drop it
into the Drawing OCR engine as an alternative detector mode.

Pipeline at inference time:
    image_rgb (any size)
        │
        ▼
   preprocess: resize so longest side == TARGET_SIZE, keep aspect ratio,
              pad with white to TARGET_SIZE × TARGET_SIZE
        │
        ▼
   CRAFT_Lite forward → (1, 2, TARGET_SIZE/2, TARGET_SIZE/2)
        │
        ▼
   post-process:
     • text_score = region_heatmap > text_threshold
     • link_score = affinity_heatmap > link_threshold
     • combined = text_score OR link_score (text + bridges)
     • connected components on combined mask
     • each component is one "word" / "line"
     • compute axis-aligned bbox for each component
     • map bboxes back to original-image coords (un-resize + un-pad)
        │
        ▼
   list of (x_min, y_min, x_max, y_max) — same format the rest of the
   Drawing OCR pipeline expects from a detector
"""

from __future__ import annotations

import os
import threading
from typing import List, Tuple

import numpy as np
import torch

from .craft_model import build_model


# CRAFT_Lite was trained at 512×512; running inference at this size keeps
# the model in-distribution.  Output stride is 2 → heatmap is 256×256.
TARGET_SIZE = 512


# ---------------------------------------------------------------------------
# Lazy singleton loader — same pattern as the recogniser engine, so the
# model is only built once and re-used across calls (and across detection
# modes within a single OCR invocation).
# ---------------------------------------------------------------------------
_MODEL = None
_DEVICE: torch.device | None = None
_LOCK = threading.Lock()


def _checkpoint_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "models", "craft_lite.pt")


def is_available() -> bool:
    """True iff a trained CRAFT_Lite checkpoint exists on disk."""
    return os.path.isfile(_checkpoint_path())


def _get_model(prefer_cuda: bool = True):
    """Build + cache the CRAFT_Lite model.  Thread-safe."""
    global _MODEL, _DEVICE
    with _LOCK:
        target = torch.device(
            "cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu"
        )
        if _MODEL is not None and _DEVICE == target:
            return _MODEL
        path = _checkpoint_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"CRAFT_Lite checkpoint not found: {path}\n"
                "Train with:  python -m drawing_ocr.craft_train"
            )
        ckpt = torch.load(path, map_location=target, weights_only=False)
        model = build_model(target, base_ch=ckpt.get("base_ch", 32))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _MODEL = model
        _DEVICE = target
        return _MODEL


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
def _preprocess(image_rgb: np.ndarray,
                target_size: int = TARGET_SIZE
                ) -> Tuple[torch.Tensor, float, int, int]:
    """Resize → pad → tensor.  Returns (tensor, scale_factor, new_w, new_h).

    `scale_factor` is the multiplier applied to the original image.
    `new_w`, `new_h` are the resized dimensions (without padding) — used
    later to clip bboxes that landed in the padded region.
    """
    h, w = image_rgb.shape[:2]
    longest = max(h, w)
    scale = target_size / max(longest, 1)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    try:
        import cv2
        resized = cv2.resize(image_rgb, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        resized = np.asarray(
            Image.fromarray(image_rgb).resize((new_w, new_h), Image.BILINEAR)
        )

    # Pad to target_size × target_size with WHITE (matches training synth
    # which had white background)
    canvas = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    arr = canvas.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, scale, new_w, new_h


# ---------------------------------------------------------------------------
# Post-processing: heatmap → bounding boxes
# ---------------------------------------------------------------------------
def _heatmap_to_boxes(
    region: np.ndarray,
    affinity: np.ndarray,
    *,
    text_threshold: float = 0.5,
    link_threshold: float = 0.3,
    low_text: float = 0.3,
    min_size: int = 8,
) -> List[Tuple[int, int, int, int]]:
    """Standard CRAFT post-processing on stride-2 heatmaps.

    Returns axis-aligned bboxes in HEATMAP coordinates (which are 1/2 the
    input-image coordinates because of the stride-2 output).  Caller is
    responsible for the ×2 + unscale step to get original-image coords.
    """
    try:
        import cv2
    except ImportError:
        # No OpenCV → can't do connected components; return nothing
        return []

    # Combine region + affinity into a single binary mask.  Region tells
    # us "this is text"; affinity tells us "these two letters belong
    # together" → linking them into the same connected component groups
    # adjacent chars into a single word/line bbox.
    text_score = (region >= low_text).astype(np.uint8)
    link_score = (affinity >= link_threshold).astype(np.uint8)
    combined = np.clip(text_score + link_score, 0, 1).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined, connectivity=4,
    )

    boxes: List[Tuple[int, int, int, int]] = []
    for k in range(1, n_labels):
        x, y, w, h, area = stats[k]
        if area < min_size * min_size:
            continue
        # Validate: the connected component must contain at least one
        # pixel that exceeds `text_threshold` in REGION (not just
        # affinity).  This rejects pure-affinity blobs that don't
        # correspond to real text.
        mask = (labels == k)
        if region[mask].max() < text_threshold:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def _boxes_to_original_coords(
    boxes_heatmap: List[Tuple[int, int, int, int]],
    *,
    scale: float,
    new_w: int,
    new_h: int,
    orig_h: int,
    orig_w: int,
    heatmap_stride: int = 2,
    padding: int = 2,
) -> List[Tuple[int, int, int, int]]:
    """Heatmap-coord bboxes → original-image-coord bboxes.

    Steps:
      1. Multiply by heatmap_stride to get input-image (padded) coords
      2. Reject boxes that fall outside the un-padded region (new_w × new_h)
      3. Divide by `scale` to get original-image coords
      4. Add a small padding so the CRNN gets some bleed room around each box
      5. Clamp to original image bounds
    """
    out = []
    for (x0, y0, x1, y1) in boxes_heatmap:
        # heatmap → input image
        x0 *= heatmap_stride; y0 *= heatmap_stride
        x1 *= heatmap_stride; y1 *= heatmap_stride
        # Reject if entirely outside the un-padded resized image
        if x0 >= new_w or y0 >= new_h:
            continue
        # Clip to un-padded region
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(new_w, x1); y1 = min(new_h, y1)
        # Resized → original
        ox0 = x0 / scale; oy0 = y0 / scale
        ox1 = x1 / scale; oy1 = y1 / scale
        # Add small padding then clamp
        ox0 = max(0, int(ox0 - padding))
        oy0 = max(0, int(oy0 - padding))
        ox1 = min(orig_w, int(ox1 + padding))
        oy1 = min(orig_h, int(oy1 + padding))
        if ox1 <= ox0 or oy1 <= oy0:
            continue
        out.append((ox0, oy0, ox1, oy1))
    return out


def _expand_box(
    box: Tuple[int, int, int, int],
    *,
    orig_w: int,
    orig_h: int,
    pad_ratio: float = 0.25,
    min_pad: int = 2,
) -> Tuple[int, int, int, int]:
    """Expand a tight bbox by `pad_ratio` × height on all sides, then
    clamp to image bounds.  CRAFT boxes are glyph-tight; the CRNN needs
    surrounding whitespace to recognise reliably.
    """
    x0, y0, x1, y1 = box
    h = y1 - y0
    pad = max(min_pad, int(round(h * pad_ratio)))
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(orig_w, x1 + pad),
        min(orig_h, y1 + pad),
    )


def _group_boxes_into_lines(
    boxes: List[Tuple[int, int, int, int]],
    *,
    y_overlap_ratio: float = 0.4,
) -> List[Tuple[int, int, int, int]]:
    """Merge character-level boxes into line-level boxes.

    Two boxes are considered same-line if their vertical overlap divided
    by the SMALLER box's height exceeds `y_overlap_ratio`.  Once grouped,
    each line's bbox is the union (min/max) of its members.

    This is what the rest of the Drawing OCR pipeline expects from a
    detector: one bbox per text line, fed to the CRNN as a single strip.
    """
    if not boxes:
        return []
    # Sort by y-center then x; iterate and assign to existing lines or
    # start a new one.
    rem = sorted(boxes, key=lambda b: ((b[1] + b[3]) / 2, b[0]))
    lines: List[List[Tuple[int, int, int, int]]] = []
    for b in rem:
        bh = b[3] - b[1]
        placed = False
        for line in lines:
            # Check overlap against the line's current y-range
            lx0 = min(c[0] for c in line); ly0 = min(c[1] for c in line)
            lx1 = max(c[2] for c in line); ly1 = max(c[3] for c in line)
            lh = ly1 - ly0
            inter = max(0, min(b[3], ly1) - max(b[1], ly0))
            if inter / max(min(bh, lh), 1) >= y_overlap_ratio:
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])
    out = []
    for line in lines:
        x0 = min(c[0] for c in line); y0 = min(c[1] for c in line)
        x1 = max(c[2] for c in line); y1 = max(c[3] for c in line)
        out.append((x0, y0, x1, y1))
    # Top-to-bottom ordering matches the engine's expectation
    out.sort(key=lambda b: (b[1], b[0]))
    return out


# ---------------------------------------------------------------------------
# Public detector class — mirrors `easy_reader.detect()` shape
# ---------------------------------------------------------------------------
class CraftLiteDetector:
    """Trained CRAFT_Lite as a drop-in text detector.

    Usage:
        det = CraftLiteDetector()
        boxes = det.detect(image_rgb)         # list of (x_min, y_min, x_max, y_max)
    """

    def __init__(self,
                 checkpoint_path: str | None = None,
                 device: str = "auto"):
        # Allow override; otherwise the singleton loader uses the default.
        if checkpoint_path is not None:
            # Single-instance override — bypass global cache
            self._device = self._resolve_device(device)
            ckpt = torch.load(checkpoint_path, map_location=self._device,
                              weights_only=False)
            self._model = build_model(self._device,
                                      base_ch=ckpt.get("base_ch", 32))
            self._model.load_state_dict(ckpt["state_dict"])
            self._model.eval()
        else:
            prefer_cuda = (device != "cpu")
            self._model = _get_model(prefer_cuda=prefer_cuda)
            self._device = _DEVICE

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name == "cuda":
            return torch.device("cuda")
        if name == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def detect(self,
               image_rgb: np.ndarray,
               *,
               text_threshold: float = 0.4,
               link_threshold: float = 0.2,
               low_text: float = 0.2,
               min_size: int = 4,
               group_lines: bool = True,
               ) -> List[Tuple[int, int, int, int]]:
        """Detect text regions.  Returns list of (x_min, y_min, x_max, y_max)
        axis-aligned bboxes in the ORIGINAL image's coordinate frame.

        Threshold tuning:
          * text_threshold — a region pixel must exceed this for its
            connected component to qualify as a text region (default 0.5)
          * link_threshold — affinity pixel threshold for "characters
            belong together" (default 0.3)
          * low_text — minimum region pixel value to include in
            connected-component growing (default 0.3)
          * min_size — drop components smaller than min_size × min_size
            (default 8 px in heatmap coords ≈ 16 px in input coords)

        Defaults are calibrated to match the score distribution our
        smoke-tested model produces; production training may yield
        higher peaks and we can raise the thresholds accordingly.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                "CraftLiteDetector.detect expects a (H, W, 3) RGB array"
            )

        tensor, scale, new_w, new_h = _preprocess(image_rgb)
        tensor = tensor.to(self._device, non_blocking=True)

        with torch.no_grad():
            pred = self._model(tensor)
        region = pred[0, 0].cpu().numpy()
        affinity = pred[0, 1].cpu().numpy()

        heat_boxes = _heatmap_to_boxes(
            region, affinity,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            low_text=low_text,
            min_size=min_size,
        )

        orig_h, orig_w = image_rgb.shape[:2]
        char_boxes = _boxes_to_original_coords(
            heat_boxes,
            scale=scale, new_w=new_w, new_h=new_h,
            orig_h=orig_h, orig_w=orig_w,
            heatmap_stride=2,
        )
        if group_lines:
            lines = _group_boxes_into_lines(char_boxes)
            return [
                _expand_box(b, orig_w=orig_w, orig_h=orig_h)
                for b in lines
            ]
        return char_boxes

    def detect_with_heatmaps(self, image_rgb: np.ndarray):
        """Diagnostic helper: return raw region + affinity heatmaps
        (numpy float arrays in [0,1]) alongside the boxes.  Useful for
        visualising what the model "saw" without running through the
        post-processing thresholds."""
        tensor, scale, new_w, new_h = _preprocess(image_rgb)
        tensor = tensor.to(self._device, non_blocking=True)
        with torch.no_grad():
            pred = self._model(tensor)
        region = pred[0, 0].cpu().numpy()
        affinity = pred[0, 1].cpu().numpy()
        return region, affinity, scale, new_w, new_h


# ---------------------------------------------------------------------------
# Compatibility shim for the existing engine code
# ---------------------------------------------------------------------------
def detect_boxes(image_rgb: np.ndarray, **kwargs):
    """Convenience: load the singleton detector and run detect().  Caches
    across calls so the model is only built once.

    Returns list of (x_min, y_min, x_max, y_max) bboxes.
    """
    det = CraftLiteDetector()
    return det.detect(image_rgb, **kwargs)
