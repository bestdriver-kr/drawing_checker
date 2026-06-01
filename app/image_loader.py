"""이미지 로딩: TIF(멀티페이지/CCITT/16비트 등)·PNG 등은 Pillow로, PDF는 pypdfium2로 래스터화."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageFile, ImageSequence
from PySide6.QtGui import QImage

PDF_DPI = 200  # PDF 페이지 래스터화 해상도

# 일부 스캐너/CAD가 만든 살짝 잘린 TIF도 최대한 읽도록 허용
ImageFile.LOAD_TRUNCATED_IMAGES = True

# convert("RGBA")가 직접 처리 못 하는 고비트/특수 모드(먼저 8비트로 정규화)
_HIGH_BIT_MODES = {"I;16", "I;16B", "I;16L", "I;16N", "I", "F"}


def _to_rgba(pil_img: Image.Image) -> Image.Image:
    """어떤 모드든 RGBA로 변환한다(16/32비트는 8비트로 정규화)."""
    mode = pil_img.mode
    if mode in _HIGH_BIT_MODES:
        # 16/32비트 그레이스케일 등 → 최소~최대를 0~255로 펴서 8비트 L로
        arr = np.asarray(pil_img).astype("float64")
        lo, hi = (float(arr.min()), float(arr.max())) if arr.size else (0.0, 0.0)
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        pil_img = Image.fromarray(arr.astype("uint8"), "L")
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    return pil_img


def pil_to_qimage(pil_img: Image.Image) -> QImage:
    """PIL 이미지를 detach된 QImage(RGBA8888)로 변환한다."""
    pil_img = _to_rgba(pil_img)
    data = pil_img.tobytes("raw", "RGBA")
    qimg = QImage(
        data,
        pil_img.width,
        pil_img.height,
        pil_img.width * 4,
        QImage.Format.Format_RGBA8888,
    )
    # data 버퍼는 함수 종료 후 해제되므로 copy로 내부 복사본을 만든다.
    return qimg.copy()


def load_pdf_pages(path: str, dpi: int = PDF_DPI) -> list[QImage]:
    """PDF를 페이지별로 dpi 해상도로 래스터화해 QImage 리스트로 반환한다."""
    import pypdfium2 as pdfium

    pages: list[QImage] = []
    scale = dpi / 72.0
    pdf = pdfium.PdfDocument(path)
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pil = bitmap.to_pil()
            pages.append(pil_to_qimage(pil))
    finally:
        pdf.close()
    if not pages:
        raise ValueError("PDF에서 페이지를 찾지 못했습니다.")
    return pages


def _tiff_diag(img: Image.Image) -> str:
    """실패 메시지에 덧붙일 TIF 진단 정보(모드·압축)."""
    comp = ""
    try:
        comp = str(img.tag_v2.get(259, ""))  # 259 = Compression 태그
    except Exception:  # noqa: BLE001
        pass
    return f"(mode={img.mode}, compression={comp or '?'})"


def load_pages(path: str) -> list[QImage]:
    """파일을 열어 페이지별 QImage 리스트를 반환한다.

    PDF는 pypdfium2로 래스터화, 멀티페이지 TIF는 모든 프레임을, 그 외는 단일 페이지를 반환한다.
    안 열리는 프레임이 있어도 가능한 페이지는 살리고, 전부 실패하면 원인을 담아 예외를 던진다.
    """
    if os.path.splitext(path)[1].lower() == ".pdf":
        return load_pdf_pages(path)

    pages: list[QImage] = []
    last_err: Exception | None = None
    diag = ""
    with Image.open(path) as img:
        try:
            diag = _tiff_diag(img)
        except Exception:  # noqa: BLE001
            diag = f"(mode={getattr(img, 'mode', '?')})"
        try:
            n_frames = getattr(img, "n_frames", 1)
        except Exception:  # noqa: BLE001
            n_frames = 1
        if n_frames > 1:
            for idx, frame in enumerate(ImageSequence.Iterator(img)):
                try:
                    pages.append(pil_to_qimage(frame))
                except Exception as exc:  # noqa: BLE001
                    last_err = exc  # 해당 프레임만 건너뛰고 계속
        else:
            try:
                pages.append(pil_to_qimage(img))
            except Exception as exc:  # noqa: BLE001
                last_err = exc

    if not pages:
        detail = f"\n{diag}" if diag else ""
        if last_err is not None:
            detail += f"\n원인: {last_err}"
        raise ValueError(
            "이미지를 디코딩하지 못했습니다. 지원되지 않는 압축/형식일 수 있습니다."
            + detail)
    return pages
