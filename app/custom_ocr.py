"""사용자 정의 OCR 엔진 등록 파일 (app/ocr.py가 시작 시 자동 로드).

여기서는 동봉한 도면 치수 전용 엔진 `drawing_ocr`(CRNN)을 우리 앱의
OCR 엔진으로 등록한다. 다른 엔진을 추가하려면 OcrEngine을 상속한 클래스를
만들어 register_engine(...) 하면 된다(아래 예시 클래스 참고).

엔진 계약(OcrEngine)
  - key / label
  - is_available() -> (bool, 사유)
  - detect(base: QImage, langs, min_conf) -> list[TextItem]
        rect 는 반드시 **배경 이미지의 픽셀 좌표** QRect.
"""
from __future__ import annotations

import os
import re
import sys

from PySide6.QtCore import QRect
from PySide6.QtGui import QImage

from .ocr import OcrEngine, TextItem, qimage_to_rgb, register_engine, use_gpu

# 프로젝트 루트(= drawing_ocr 패키지가 있는 곳)를 import 경로에 추가
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# variant 태그에서 업스케일 배율(x8 등)을 추출 → bbox를 원본 좌표로 환원
_SCALE_RE = re.compile(r"x([0-9.]+)")


class DrawingDimensionEngine(OcrEngine):
    """도면 치수 전용 CRNN OCR (drawing_ocr 패키지).

    숫자·공차기호(± Ø ° 등)·ISO 286 끼워맞춤 코드(H7, g6 …)에 특화.
    일반 문장보다 치수/공차 텍스트 검출에 강하다.
    """

    key = "drawing_dim"
    label = "Drawing OCR (도면 치수 전용)"
    priority = 0  # 메뉴 1순위 + 기본 엔진

    def __init__(self):
        self._ocr = None
        self._built_device = None

    def _model_path(self) -> str:
        return os.path.join(_PROJECT_ROOT, "models", "drawing_crnn.pt")

    def is_available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"torch가 설치되어 있지 않습니다: {exc}"
        try:
            import drawing_ocr  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, (
                "drawing_ocr 패키지를 불러올 수 없습니다.\n"
                "프로젝트 루트에 drawing_ocr/ 폴더가 있어야 합니다.\n"
                f"({exc})"
            )
        if not os.path.isfile(self._model_path()):
            return False, f"모델 파일이 없습니다:\n{self._model_path()}"
        return True, ""

    def _engine(self):
        device = "cuda" if use_gpu() else "cpu"  # 장치 선택 반영
        if self._ocr is None or self._built_device != device:
            from drawing_ocr import DrawingOCR
            self._ocr = DrawingOCR(model_path=self._model_path(), device=device)
            self._built_device = device
        return self._ocr

    def detect(self, base: QImage, langs, min_conf: float) -> list[TextItem]:
        rgb = qimage_to_rgb(base)
        results = self._engine().recognize_image(rgb)
        items: list[TextItem] = []
        for text, conf, bbox, tag in results:
            text = (text or "").strip()
            if not text or conf < min_conf:
                continue
            # bbox(4점 폴리곤)는 전처리 variant 좌표 → 배율로 나눠 원본 좌표화
            m = _SCALE_RE.search(tag or "")
            scale = float(m.group(1)) if m else 1.0
            xs = [p[0] / scale for p in bbox]
            ys = [p[1] / scale for p in bbox]
            rect = QRect(
                int(min(xs)), int(min(ys)),
                int(max(xs) - min(xs)), int(max(ys) - min(ys)),
            )
            items.append(TextItem(text, rect, float(conf)))
        return _dedup_overlapping(items)


def _iou(a: QRect, b: QRect) -> float:
    inter = a.intersected(b)
    if inter.isEmpty():
        return 0.0
    ia = inter.width() * inter.height()
    ua = a.width() * a.height() + b.width() * b.height() - ia
    return ia / ua if ua > 0 else 0.0


def _dedup_overlapping(items: list[TextItem], iou_thresh: float = 0.3) -> list[TextItem]:
    """여러 variant에서 나온 동일 위치 중복 박스를 신뢰도 높은 것 하나로 정리."""
    kept: list[TextItem] = []
    for it in sorted(items, key=lambda x: -x.conf):
        if any(_iou(it.rect, k.rect) > iou_thresh for k in kept):
            continue
        kept.append(it)
    return kept


register_engine(DrawingDimensionEngine())


# ───────────────────────── 다른 엔진을 더 추가하고 싶다면 ─────────────────────────
# class MyEngine(OcrEngine):
#     key = "myengine"
#     label = "내 OCR"
#     def is_available(self): return True, ""
#     def detect(self, base, langs, min_conf):
#         rgb = qimage_to_rgb(base)           # (h,w,3) numpy, 원본 픽셀 좌표
#         # ... 본인 엔진 호출 → 결과를 TextItem(text, QRect(...), conf)로 ...
#         return []
# register_engine(MyEngine())
