"""Horizontal-projection text-line splitter.

The CRNN recogniser is single-line by design — a tall multi-line capture
(stacked dimension + upper/lower tolerance, vertical dimension callouts,
limit-dimension pairs, etc.) collapses to nonsense when fed as one tensor.
This module finds horizontal text-line bands inside a crop and returns them
as separate sub-crops the recogniser can read one at a time.

Pipeline at inference time:
    captured region  ──▶  EasyOCR CRAFT detector  ──▶  raw bboxes
                                                            │
                                                            ▼
                                    split_box_into_lines(box) for each box
                                                            │
                                                            ▼
                                         per-line crops to CRNN

Plus a whole-image fallback (`detect_lines_on_whole`) for the case where
the CRAFT detector misses most of the text (common with very small or
heavily-aliased drawing crops).

Algorithm — classic OCR document layout analysis:
  1. Grayscale.
  2. Background-aware inversion so text pixels are POSITIVE
     (works for both light-bg and dark-bg captures).
  3. Optional binarisation via Otsu — sharp peaks, immune to anti-aliasing.
  4. Row-sum projection over the binarised image.
  5. Smooth (rolling mean) to fuse decimal points / commas with their
     parent line.
  6. Threshold-based band extraction: contiguous rows above a fraction of
     peak projection form a band; gaps below the threshold are valleys.
  7. Filter bands by minimum height (default 6 px) to drop dust-speck
     noise.
  8. Add a small padding (default 2 px) around each band so glyph
     descenders / ascenders aren't clipped.

The thresholds are conservative — when in doubt we KEEP a band rather
than drop it, because false-positive lines just get extra CRNN calls,
whereas false-negative lines vanish from the output entirely.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Tunable constants.  Exposed as module-level so tests / experiments can
# tweak without touching call sites.
# ---------------------------------------------------------------------------
DEFAULT_VALLEY_FRAC = 0.10
# Rows with smoothed projection below VALLEY_FRAC * peak are treated as
# inter-line gap.  0.10 picks up almost every visible gap on engineering
# crops while being robust to imaging noise.

DEFAULT_SMOOTH_WIN = 3
# Rolling-mean window size in PIXELS.  3 fuses a stray decimal dot with
# the line above/below it; larger windows risk merging actually-separate
# rows.

DEFAULT_MIN_LINE_H = 6
# Minimum band height in pixels at NATIVE capture resolution.  Drawings
# below this height are usually single-digit fragments mis-segmented from
# the detector and not real text lines.

DEFAULT_BAND_PAD = 2
# Pixels of padding above + below each detected band so descenders /
# ascenders survive the crop.

DEFAULT_MULTILINE_ASPECT = 0.50
# A bbox whose height >= aspect * width is a candidate for line-splitting.
# Smaller values (more aggressive) → splitter inspects more boxes.
# 0.5 is the sweet spot — captures with two or more stacked lines have
# height comparable to width, while a horizontal "Ø50 ±0.05" single line
# is much wider than tall and skips splitting.


def _project_rows(gray: np.ndarray) -> np.ndarray:
    """Return per-row projection (sum of foreground pixels per row) after
    background-aware inversion + Otsu binarisation.

    The classic trick: binarise so foreground (text) is 255 and background
    is 0, then ROW-SUM gives strong peaks at text bands and clean zeros
    in valleys.  CLAHE first to stabilise the threshold under uneven
    illumination."""
    try:
        import cv2
    except ImportError:
        # Fallback without OpenCV — direct intensity inversion + sum
        if gray.mean() > 128:
            fg = 255 - gray
        else:
            fg = gray
        return fg.astype(np.float32).sum(axis=1)

    # CLAHE for stable threshold under uneven backgrounds.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    # Background-aware inversion so the Otsu threshold finds text.
    if eq.mean() > 128:
        inv = 255 - eq
    else:
        inv = eq
    _, bw = cv2.threshold(inv, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw.astype(np.float32).sum(axis=1)


def _smooth(arr: np.ndarray, win: int) -> np.ndarray:
    """Rolling-mean smoothing — uses `np.convolve` for clarity at the cost
    of two extra allocations.  Edge handling: reflect-padded so the first
    and last rows aren't artificially attenuated."""
    if win <= 1 or arr.size <= win:
        return arr.astype(np.float32)
    pad = win // 2
    padded = np.concatenate(
        [arr[pad - 1::-1].astype(np.float32),
         arr.astype(np.float32),
         arr[-1:-pad - 1:-1].astype(np.float32)]
    )
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(padded, kernel, mode="valid")


def find_text_line_bands(
    img_rgb: np.ndarray,
    *,
    valley_frac: float = DEFAULT_VALLEY_FRAC,
    smooth_win: int = DEFAULT_SMOOTH_WIN,
    min_line_h: int = DEFAULT_MIN_LINE_H,
    band_pad: int = DEFAULT_BAND_PAD,
) -> List[Tuple[int, int]]:
    """Locate horizontal text-line bands in `img_rgb`.

    Returns a list of (y_start, y_end) pairs (inclusive starts, exclusive
    ends in NumPy convention).  Y is the row index in the input image.
    Empty list means nothing detected.

    The function is conservative: if the projection is nearly flat (single
    block of text or pure noise), it returns one band covering the whole
    image rather than splitting at arbitrary noise valleys.
    """
    if img_rgb.ndim == 2:
        gray = img_rgb
    elif img_rgb.ndim == 3:
        # PIL→numpy gives RGB; mean is fine for grayscale projection.
        gray = img_rgb.mean(axis=2).astype(np.uint8)
    else:
        return []

    h = gray.shape[0]
    if h <= min_line_h * 2:
        # Too small to host two lines — single-band shortcut.
        return [(0, h)]

    proj = _project_rows(gray)
    if proj.max() <= 1e-6:
        return []  # Pure background, no text.

    proj = _smooth(proj, smooth_win)
    peak = proj.max()
    threshold = peak * valley_frac

    # Extract contiguous bands where proj > threshold.
    above = proj > threshold
    bands: List[Tuple[int, int]] = []
    in_band = False
    band_start = 0
    for i, hot in enumerate(above):
        if hot and not in_band:
            in_band = True
            band_start = i
        elif (not hot) and in_band:
            in_band = False
            if i - band_start >= min_line_h:
                bands.append((band_start, i))
    if in_band:
        if h - band_start >= min_line_h:
            bands.append((band_start, h))

    # If the splitter produced just one band that covers most of the
    # image, the input is single-line — return [(0, h)] so downstream
    # code uses the natural bbox rather than the slightly-trimmed one.
    if len(bands) == 1:
        bs, be = bands[0]
        if (be - bs) >= 0.85 * h:
            return [(0, h)]

    # Pad each band, clamp to image bounds.
    padded = []
    for bs, be in bands:
        padded.append((max(0, bs - band_pad), min(h, be + band_pad)))
    return padded if padded else [(0, h)]


def split_box_into_lines(
    img_rgb: np.ndarray,
    x_min: int, y_min: int, x_max: int, y_max: int,
    *,
    multiline_aspect: float = DEFAULT_MULTILINE_ASPECT,
    **kwargs,
) -> List[Tuple[int, int, int, int]]:
    """Given a CRAFT/EasyOCR detection box, return a list of (x0, y0, x1,
    y1) sub-boxes — one per text line found inside.

    For visually single-line boxes (very wide compared to tall) the
    function returns the original box unchanged.  Only "tall" boxes are
    inspected for internal line splitting.

    `kwargs` are forwarded to `find_text_line_bands` (valley_frac etc.).
    """
    # Clamp + sanity
    h_img, w_img = img_rgb.shape[:2]
    x_min = max(0, int(x_min)); y_min = max(0, int(y_min))
    x_max = min(w_img, int(x_max)); y_max = min(h_img, int(y_max))
    if x_max <= x_min or y_max <= y_min:
        return []

    box_w = x_max - x_min
    box_h = y_max - y_min
    if box_w == 0 or box_h / box_w < multiline_aspect:
        # Wider than (1/aspect) tall — almost certainly single line.
        return [(x_min, y_min, x_max, y_max)]

    sub = img_rgb[y_min:y_max, x_min:x_max]
    bands = find_text_line_bands(sub, **kwargs)
    if len(bands) <= 1:
        return [(x_min, y_min, x_max, y_max)]

    out = []
    for bs, be in bands:
        # Y is relative to the cropped sub-image — shift back to global.
        out.append((x_min, y_min + bs, x_max, y_min + be))
    return out


def detect_lines_on_whole(
    img_rgb: np.ndarray,
    **kwargs,
) -> List[Tuple[int, int, int, int]]:
    """Fallback when the CRAFT detector misses everything: scan the whole
    image with the projection splitter and return each band as a full-width
    bbox.  Returned format matches `split_box_into_lines`.
    """
    h, w = img_rgb.shape[:2]
    bands = find_text_line_bands(img_rgb, **kwargs)
    if not bands:
        return [(0, 0, w, h)]
    return [(0, bs, w, be) for bs, be in bands]
