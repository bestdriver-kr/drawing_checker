"""이미지 로딩: TIF(멀티페이지/CCITT)·PNG 등은 Pillow로, PDF는 pypdfium2로 래스터화."""
from __future__ import annotations

import os

from PIL import Image, ImageSequence
from PySide6.QtGui import QImage

PDF_DPI = 200  # PDF 페이지 래스터화 해상도


def pil_to_qimage(pil_img: Image.Image) -> QImage:
    """PIL 이미지를 detach된 QImage(RGBA8888)로 변환한다."""
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
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


def load_pages(path: str) -> list[QImage]:
    """파일을 열어 페이지별 QImage 리스트를 반환한다.

    PDF는 pypdfium2로 래스터화, 멀티페이지 TIF는 모든 프레임을, 그 외는 단일 페이지를 반환한다.
    """
    if os.path.splitext(path)[1].lower() == ".pdf":
        return load_pdf_pages(path)

    pages: list[QImage] = []
    with Image.open(path) as img:
        n_frames = getattr(img, "n_frames", 1)
        if n_frames > 1:
            for frame in ImageSequence.Iterator(img):
                pages.append(pil_to_qimage(frame))
        else:
            pages.append(pil_to_qimage(img))
    if not pages:
        raise ValueError("이미지에서 페이지를 찾지 못했습니다.")
    return pages
