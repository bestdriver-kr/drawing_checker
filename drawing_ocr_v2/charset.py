"""Character set for engineering-drawing OCR.

Scope (per request): digits + tolerance/dimension symbols only.  Letter-based
ISO 286 fit codes (H7, g6, ...) are intentionally excluded — they require an
alphabet recogniser, which defeats the point of a narrow domain model.

Layout:
  * index 0 is CTC blank
  * indices 1..N-1 are visible characters
  * lookup is exact: visually-similar variants (Ø vs ⌀ vs Φ) are normalised
    to a single canonical form at encode time so the model only needs to
    learn one representation per phoneme.
"""

# Canonical character set.  Order is fixed — the model checkpoint depends on
# these indices.  Append new chars at the end; never reorder existing ones.
#
# v2 (2026-05-29) extends v1 with the FULL Latin alphabet (a-z + A-Z minus
# the two letters already present from v1: lowercase 'x' and uppercase 'R').
# This is required to recognise ISO 286 fit codes — H7, g6, f7, js9, etc.
# — and thread designations like M8.  Adding letters changes NUM_CLASSES
# from 22 to 72 so the v1 checkpoint is INVALID for v2; full retraining
# is required.  The v1 binary is kept at models/drawing_crnn_v1_22cls.pt
# for fallback.
_VISIBLE = (
    "0123456789"   # digits
    "+-±"          # signs
    ".,/"          # separators
    "Ø"            # diameter symbol (canonical; ⌀/Φ map here)
    "R"            # radius prefix (also part of A-Z but kept distinct from v1)
    "°"            # degree
    "x"            # thread/multiplier ("M8x1.25", "3x45°") — distinct from 'X'
    " "            # space — separates tokens in stacked-on-one-line text
    # ── v2 alphabet extension ────────────────────────────────────────────
    # Lowercase a-z, skipping 'x' (already present, index above) and 'r'
    # (NOT skipped — uppercase R is at the v1 index, lowercase r is new).
    "abcdefghijklmnopqrstuvwyz"   # 25 lowercase: a-z minus x
    # Uppercase A-Z, skipping 'R' (already present at v1 index).
    "ABCDEFGHIJKLMNOPQSTUVWXYZ"   # 25 uppercase: A-Z minus R
)

# CHARSET is a list (not a string) so the blank token occupies exactly one
# index instead of being split into seven characters of "<BLANK>".  Visible
# chars are at indices 1..N-1.
_BLANK_TOKEN = "<BLANK>"
CHARSET = [_BLANK_TOKEN] + list(_VISIBLE)
BLANK_IDX = 0
NUM_CLASSES = len(CHARSET)

# Normalisation: collapse visually-equivalent Unicode variants onto the
# canonical character so encode() can do a straight lookup.
_NORMALIZE = {
    "⌀": "Ø",  # U+2300 DIAMETER SIGN
    "Φ": "Ø",  # U+03A6 GREEK CAPITAL PHI (often used as diameter on Asian drawings)
    "∅": "Ø",  # U+2205 EMPTY SET (yet another diameter substitute)
    "ø": "Ø",  # lowercase variant
    "−": "-",  # U+2212 MINUS SIGN → ASCII hyphen-minus
    "–": "-",  # en dash
    "—": "-",  # em dash
    "／": "/",  # full-width slash
    "　": " ",  # full-width space
    "X": "x",  # capital X used as multiplier
}

# Forward + reverse maps for encode/decode.
_CHAR_TO_IDX = {ch: i for i, ch in enumerate(CHARSET)}
_IDX_TO_CHAR = {i: ch for i, ch in enumerate(CHARSET)}


def normalize(text: str) -> str:
    """Map Unicode lookalikes to canonical characters."""
    return "".join(_NORMALIZE.get(ch, ch) for ch in text)


def encode(text: str) -> list[int]:
    """Convert a (normalised) string to a list of class indices.

    Raises KeyError on out-of-charset characters — callers should filter
    or normalise before encoding.
    """
    text = normalize(text)
    return [_CHAR_TO_IDX[ch] for ch in text]


def encode_safe(text: str) -> list[int]:
    """Encode but silently drop out-of-charset characters."""
    text = normalize(text)
    return [_CHAR_TO_IDX[ch] for ch in text if ch in _CHAR_TO_IDX]


def decode(indices) -> str:
    """Inverse of encode for a list/iterable of class indices.  Skips blanks."""
    return "".join(_IDX_TO_CHAR[i] for i in indices if i != BLANK_IDX)


def ctc_greedy_decode(logits) -> str:
    """Collapse a (T, C) logit tensor or argmax sequence to a string.

    Standard CTC greedy decoding: argmax per timestep, then collapse runs
    of identical labels, then drop blanks.
    """
    import torch
    if hasattr(logits, "argmax"):
        # logits is (T, C) tensor — pick best class per timestep
        idx_seq = logits.argmax(dim=-1).tolist()
    else:
        idx_seq = list(logits)
    out = []
    prev = -1
    for i in idx_seq:
        if i != prev and i != BLANK_IDX:
            out.append(_IDX_TO_CHAR[i])
        prev = i
    return "".join(out)
