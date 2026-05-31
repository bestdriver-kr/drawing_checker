"""Image preprocessing pipeline — extracted from the full Tolerance OCR
project so this standalone package has no dependency on its parent.

What it does
============
Engineering-drawing OCR has a few specific imaging challenges that vanilla
preprocessing doesn't address well:

  * Tiny text (10-20 px tall) — needs upscaling without smudging strokes
  * Thin "+", "-", ".", "," glyphs that anti-aliasing removes
  * Mixed dark-on-light and light-on-dark CAD backgrounds
  * Stroke-width variation between dimensions on the same drawing

`build_variants(img_rgb)` produces 5-7 differently-preprocessed copies of
one capture.  Each is fed to the recogniser, and the pooled results are
returned with their confidence — the recogniser picks whichever variant
read most cleanly.

Public API
==========
  build_variants(image_rgb, *, scale=None, bg_mode='auto') -> list[(tag, img, scale)]
  maybe_invert(image_rgb, *, bg_mode='auto') -> np.ndarray
  lanczos_upscale(image_rgb, scale) -> np.ndarray
  unsharp(image_rgb, amount=0.6) -> np.ndarray
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Atomic operations
# ---------------------------------------------------------------------------
def lanczos_upscale(img: np.ndarray, scale: float) -> np.ndarray:
    """LANCZOS4 upscale.  Falls back to nearest-neighbour when OpenCV is
    unavailable so the package keeps working on minimal installs."""
    if scale <= 1.0:
        return img.copy()
    try:
        import cv2
        return cv2.resize(img, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_LANCZOS4)
    except ImportError:
        s = max(1, int(round(scale)))
        return np.kron(img, np.ones((s, s, 1), dtype=img.dtype))


def unsharp(img: np.ndarray, amount: float = 0.6) -> np.ndarray:
    """Light unsharp-mask sharpening — recovers edges blurred by upscaling.
    No-op when OpenCV is missing."""
    try:
        import cv2
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        out = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        return np.clip(out, 0, 255).astype(np.uint8)
    except ImportError:
        return img


def maybe_invert(img: np.ndarray, *, bg_mode: str = "auto") -> np.ndarray:
    """Invert dark-on-light backgrounds so text becomes black-on-white,
    which matches the recogniser's training distribution.

    bg_mode:
      'auto'  — invert when mean intensity is below 110 (dark CAD)
      'light' — never invert
      'dark'  — always invert
    """
    if bg_mode == "dark":
        return 255 - img
    if bg_mode == "light":
        return img
    return 255 - img if img.mean() < 110 else img


# ---------------------------------------------------------------------------
# Binarisation variants — different paths emphasise different glyph features.
# ---------------------------------------------------------------------------
def _binarize_otsu(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE + Otsu global threshold + stroke dilation.  Best when
    illumination is even across the crop."""
    try:
        import cv2
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, bw = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if (bw == 0).sum() > (bw == 255).sum():
            bw = 255 - bw
        kernel = np.ones((2, 2), np.uint8)
        bw = cv2.dilate(255 - bw, kernel, iterations=1)
        bw = 255 - bw
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    except ImportError:
        return img_rgb


def _binarize_adaptive(img_rgb: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian threshold — local windows, better for uneven
    backgrounds (hatching, leader lines, faint gradients)."""
    try:
        import cv2
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        h, w = gray.shape[:2]
        block = max(11, (min(h, w) // 20) | 1)
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, 7,
        )
        if (bw == 0).sum() > (bw == 255).sum():
            bw = 255 - bw
        kernel = np.ones((2, 2), np.uint8)
        bw = cv2.dilate(255 - bw, kernel, iterations=1)
        bw = 255 - bw
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    except ImportError:
        return img_rgb


def _emphasize_horizontal(img_rgb: np.ndarray) -> np.ndarray:
    """Horizontal-stroke emphasis: thickens "-", ".", and fraction bars so
    the detector + recogniser don't drop them.  Vertical strokes (1, 7, |)
    are left alone."""
    try:
        import cv2
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray if gray.mean() > 128 else gray.copy()
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        inv = clahe.apply(inv)
        _, bw = cv2.threshold(inv, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
        bw = cv2.dilate(bw, h_kernel, iterations=1)
        bw = 255 - bw
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    except ImportError:
        return img_rgb


# ---------------------------------------------------------------------------
# Multi-variant pipeline
# ---------------------------------------------------------------------------
def build_variants(
    image_rgb: np.ndarray,
    *,
    scale: float | None = None,
    bg_mode: str = "auto",
) -> List[Tuple[str, np.ndarray, float]]:
    """Produce complementary preprocessed variants for one capture.

    Returns a list of (tag, image, upscale_factor) triples.  The caller
    runs OCR on each variant and pools/dedupes the results.

    scale: forced upscale multiplier.  When None, picked automatically
        based on the input's short side — small captures need 6-8×,
        large ones don't need upscaling at all.

    bg_mode: 'auto' (default) | 'light' | 'dark' — controls inversion.
    """
    h, w = image_rgb.shape[:2]
    short = min(h, w)

    if scale is None:
        if   short < 60:  scale = 8.0
        elif short < 120: scale = 6.0
        elif short < 240: scale = 4.0
        elif short < 480: scale = 2.5
        else:             scale = 1.5 if short < 720 else 1.0

    # Variant 1: LANCZOS upscale + sharpen — proven all-rounder baseline.
    lz = lanczos_upscale(image_rgb, scale) if scale != 1.0 else image_rgb.copy()
    lz = maybe_invert(lz, bg_mode=bg_mode)
    lz_sharp = unsharp(lz, amount=0.6)

    variants: List[Tuple[str, np.ndarray, float]] = []
    variants.append((f"lanczos-x{scale:g}", lz_sharp, scale))

    # Variant 2: global Otsu binarisation on the upscaled image.
    variants.append((f"bin-x{scale:g}", _binarize_otsu(lz), scale))

    # Variant 3: adaptive Gaussian threshold — handles uneven backgrounds.
    variants.append((f"adapt-x{scale:g}", _binarize_adaptive(lz), scale))

    # Variant 4: horizontal-stroke emphasis — rescues thin minus signs and
    # decimal points.
    variants.append((f"hbar-x{scale:g}", _emphasize_horizontal(lz), scale))

    # Variant 5 (medium captures only): extra 12× LANCZOS, useful for the
    # 40-200 px short-side regime where the nominal is readable at 6× but
    # the tolerance text needs more pixels.
    if 40 < short <= 200:
        big = lanczos_upscale(image_rgb, 12.0)
        big = maybe_invert(big, bg_mode=bg_mode)
        big = unsharp(big, amount=0.7)
        variants.append(("lanczos-x12", big, 12.0))

    return variants
