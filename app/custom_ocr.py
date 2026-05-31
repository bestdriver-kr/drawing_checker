"""사용자 정의 OCR 엔진 등록 (app/ocr.py가 시작 시 자동 로드).

동봉한 도면 치수 전용 엔진 `drawing_ocr_v2`를 두 가지 인식기로 등록한다:
  - Drawing OCR (TrOCR·정확): 파인튜닝 TrOCR, 정확도 높음, 느림(GPU 권장)
  - Drawing OCR (CRNN·빠름): 자체 CRNN, 가볍고 빠름(CPU도 OK), 정확도 보통
둘 다 검출은 자체 Projection/CRAFT_Lite를 쓰며 EasyOCR에 의존하지 않는다.
"""
from __future__ import annotations

import os
import sys

from PySide6.QtCore import QRect
from PySide6.QtGui import QImage

from .ocr import (
    OcrEngine,
    TextItem,
    cuda_available,
    qimage_to_rgb,
    register_engine,
    use_gpu,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# PyInstaller로 묶인 경우 동봉 데이터는 sys._MEIPASS 아래.
_DATA_ROOT = getattr(sys, "_MEIPASS", _PROJECT_ROOT)


def _models_dir() -> str:
    for base in (_DATA_ROOT, _PROJECT_ROOT):
        d = os.path.join(base, "models")
        if os.path.isdir(d):
            return d
    return os.path.join(_DATA_ROOT, "models")


class DrawingDimensionEngine(OcrEngine):
    """도면 치수 전용 OCR v2 (자체 개발). recognizer로 TrOCR/CRNN 선택."""

    def __init__(self, recognizer: str, key: str, label: str, priority: int,
                 force_device: str, trocr_model: str | None = None):
        self.recognizer = recognizer       # "trocr" | "crnn"
        self.force_device = force_device   # "gpu" | "cpu" | "auto"(메뉴 따름)
        # trocr_model: None=동봉 파인튜닝 모델, 그 외=경로/HF ID(예: 범용 기본 모델)
        self.trocr_model = trocr_model
        self.key = key
        self.label = label
        self.priority = priority
        self._ocr = None
        self._built_device = None

    def is_available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"torch가 설치되어 있지 않습니다: {exc}"
        try:
            import drawing_ocr_v2  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"drawing_ocr_v2 패키지를 불러올 수 없습니다: {exc}"
        md = _models_dir()
        if self.recognizer == "trocr":
            try:
                import transformers  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                return False, f"transformers가 설치되어 있지 않습니다: {exc}"
            # 동봉 파인튜닝 모델을 쓰는 경우에만 폴더 확인(범용 모델은 캐시/허브에서 로드)
            if self.trocr_model is None and not os.path.isdir(
                    os.path.join(md, "trocr_finetuned")):
                return False, f"TrOCR 모델 폴더가 없습니다:\n{md}/trocr_finetuned"
        else:  # crnn
            if not os.path.isfile(os.path.join(md, "drawing_crnn.pt")):
                return False, f"CRNN 모델이 없습니다:\n{md}/drawing_crnn.pt"
        return True, ""

    def _engine(self):
        if self.force_device == "gpu":
            device = "cuda" if cuda_available() else "cpu"
        elif self.force_device == "auto":
            device = "cuda" if use_gpu() else "cpu"  # 연산 장치 메뉴 따름
        else:  # "cpu"
            device = "cpu"
        if self._ocr is None or self._built_device != device:
            from drawing_ocr_v2 import OCR
            md = _models_dir()
            trocr_path = self.trocr_model or os.path.join(md, "trocr_finetuned")
            self._ocr = OCR(
                detector="projection",
                recognizer=self.recognizer,
                trocr_model_path=trocr_path,
                crnn_model_path=os.path.join(md, "drawing_crnn.pt"),
                craft_lite_path=os.path.join(md, "craft_lite.pt"),
                device=device,
            )
            self._built_device = device
        return self._ocr

    def detect(self, base: QImage, langs, min_conf: float) -> list[TextItem]:
        rgb = qimage_to_rgb(base)
        items: list[TextItem] = []
        for text, conf, bbox in self._engine().recognize(rgb):
            text = (text or "").strip()
            if not text or conf < min_conf or not bbox:
                continue  # 회전 패스 등 좌표 없는 결과는 제외
            x0, y0, x1, y1 = bbox  # (x_min, y_min, x_max, y_max)
            items.append(TextItem(
                text, QRect(int(x0), int(y0), int(x1 - x0), int(y1 - y0)), float(conf)))
        return items


register_engine(DrawingDimensionEngine(
    "trocr", "drawing_trocr", "Drawing OCR (자체개발·타입1·정확)", 0, "gpu"))
register_engine(DrawingDimensionEngine(
    "crnn", "drawing_crnn", "Drawing OCR (자체개발·타입2·빠름)", 1, "auto"))
register_engine(DrawingDimensionEngine(
    "trocr", "trocr_generic", "TrOCR (범용 인쇄체·GPU)", 2, "gpu",
    trocr_model="microsoft/trocr-small-printed"))
