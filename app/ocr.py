"""OCR 기반 마킹 검사 + OCR 엔진 레지스트리.

도면 배경에서 텍스트 영역을 검출하고, 그 위에 마커(펜/형광펜)가 없는 항목을 찾는다.
- "마킹됨" 판정: 텍스트 박스 영역에 보이는 주석 레이어의 픽셀이 조금이라도 있으면 마킹된 것으로 본다.

OCR 엔진 추가 방법
------------------
1) `OcrEngine`을 상속해 클래스를 만든다(아래 EasyOcrEngine/TesseractEngine 참고).
   - key      : 고유 문자열 ID
   - label    : 메뉴에 보일 이름
   - is_available() -> (bool, 사유)  : 사용 가능 여부
   - detect(base, langs, min_conf) -> list[TextItem]
       base    : QImage (배경 도면, 원본 픽셀 좌표)
       반환     : 검출된 텍스트들. rect는 **이미지 픽셀 좌표**의 QRect.
   - 이미지를 numpy로 받고 싶으면 `qimage_to_rgb(base)` 헬퍼를 쓰면 된다.
2) `register_engine(MyEngine())` 으로 등록한다.
   - 보통은 `app/custom_ocr.py`(자동 로드됨)에서 등록한다.
3) 끝. `검사 → OCR 엔진` 메뉴에 자동으로 나타난다.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QImage

from .layer import Page


@dataclass
class TextItem:
    text: str
    rect: QRect      # 이미지 픽셀 좌표
    conf: float      # 0.0~1.0


# ---------------------------------------------------------------- 연산 장치 설정
DEVICE_AUTO = "auto"   # CUDA 있으면 GPU, 없으면 CPU
DEVICE_GPU = "gpu"     # GPU 강제(없으면 CPU로 폴백)
DEVICE_CPU = "cpu"     # CPU 강제

_device_pref = DEVICE_AUTO


def set_device(pref: str):
    """OCR 연산 장치 선호도 설정(auto/gpu/cpu). 다음 검출부터 적용."""
    global _device_pref
    if pref in (DEVICE_AUTO, DEVICE_GPU, DEVICE_CPU):
        _device_pref = pref


def get_device() -> str:
    return _device_pref


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def use_gpu() -> bool:
    """현재 선호도+가용성으로 GPU를 쓸지 결정(CPU 강제 시 False)."""
    if _device_pref == DEVICE_CPU:
        return False
    return cuda_available()  # auto/gpu 모두 CUDA 있을 때만 GPU


# ---------------------------------------------------------------- 이미지 헬퍼
def qimage_to_rgb(qimg: QImage) -> np.ndarray:
    """QImage → (h, w, 3) uint8 RGB ndarray."""
    img = qimg.convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(h, bpl)
    return arr[:, : w * 3].reshape(h, w, 3).copy()


def _alpha_array(qimg: QImage) -> np.ndarray:
    img = qimg.convertToFormat(QImage.Format.Format_ARGB32)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(h, bpl)
    return arr[:, : w * 4].reshape(h, w, 4)[:, :, 3].copy()


def qimage_to_png_bytes(qimg: QImage) -> bytes:
    """QImage → PNG 바이트(Windows OCR 등 외부 디코더 입력용)."""
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    qimg.save(buf, "PNG")
    buf.close()
    return bytes(ba)


# ---------------------------------------------------------------- 엔진 인터페이스
class OcrEngine:
    """OCR 엔진이 구현해야 하는 인터페이스."""

    key: str = ""
    label: str = ""
    priority: int = 50  # 작을수록 메뉴에서 먼저(기본값 후보 우선순위도 동일)

    def is_available(self) -> tuple[bool, str]:
        """사용 가능하면 (True, ''), 아니면 (False, 사유)."""
        return True, ""

    def detect(self, base: QImage, langs: tuple[str, ...],
               min_conf: float) -> list[TextItem]:
        raise NotImplementedError


_ENGINES: dict[str, OcrEngine] = {}


def register_engine(engine: OcrEngine):
    """엔진을 등록한다(이미 같은 key가 있으면 덮어씀)."""
    if not engine.key:
        raise ValueError("엔진 key가 비어 있습니다.")
    _ENGINES[engine.key] = engine


def list_engines() -> list[tuple[str, str]]:
    """(키, 표시이름) 목록 — priority 오름차순(같으면 등록 순서)."""
    engines = sorted(_ENGINES.values(), key=lambda e: e.priority)  # stable
    return [(e.key, e.label) for e in engines]


DEFAULT_ENGINE_KEY = "rapidocr"  # 시작 시 기본 선택 엔진


def default_engine() -> str:
    """기본 엔진 키. 지정 엔진이 등록돼 있으면 그것, 아니면 priority 최상위."""
    if DEFAULT_ENGINE_KEY in _ENGINES:
        return DEFAULT_ENGINE_KEY
    engines = list_engines()
    return engines[0][0] if engines else ENGINE_EASYOCR


def engine_label(key: str) -> str:
    eng = _ENGINES.get(key)
    return eng.label if eng else key


def engine_available(key: str) -> tuple[bool, str]:
    eng = _ENGINES.get(key)
    if eng is None:
        return False, f"알 수 없는 엔진: {key}"
    try:
        return eng.is_available()
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def detect_text_boxes(
    base: QImage,
    engine: str,
    langs: tuple[str, ...] = ("en",),
    min_conf: float = 0.3,
) -> list[TextItem]:
    """선택한 엔진으로 텍스트 박스를 검출한다."""
    eng = _ENGINES.get(engine)
    if eng is None:
        raise ValueError(f"등록되지 않은 OCR 엔진: {engine}")
    return eng.detect(base, langs, min_conf)


# ---------------------------------------------------------------- 내장 엔진: EasyOCR
class EasyOcrEngine(OcrEngine):
    key = "easyocr"
    label = "EasyOCR (딥러닝·정확)"

    def __init__(self):
        self._reader = None
        self._reader_key: tuple | None = None  # (langs, gpu)

    def is_available(self) -> tuple[bool, str]:
        try:
            import easyocr  # noqa: F401
            return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, f"easyocr 모듈을 불러올 수 없습니다: {exc}"

    @staticmethod
    def _bundled_model_dir():
        """PyInstaller로 동봉된 EasyOCR 모델 폴더(있으면). 없으면 None."""
        base = getattr(sys, "_MEIPASS", None)
        if base:
            d = os.path.join(base, "easyocr_model")
            if os.path.isdir(d):
                return d
        return None

    def _reader_for(self, langs):
        gpu = use_gpu()
        key = (langs, gpu)
        if self._reader is None or self._reader_key != key:
            import easyocr  # 무거운 import는 지연
            model_dir = self._bundled_model_dir()
            if model_dir:
                # 동봉 모델 사용 + 다운로드 비활성(오프라인/보안 친화)
                self._reader = easyocr.Reader(
                    list(langs), gpu=gpu,
                    model_storage_directory=model_dir, download_enabled=False)
            else:
                self._reader = easyocr.Reader(list(langs), gpu=gpu)
            self._reader_key = key
        return self._reader

    def detect(self, base, langs, min_conf):
        reader = self._reader_for(langs)
        results = reader.readtext(qimage_to_rgb(base))
        items: list[TextItem] = []
        for bbox, text, conf in results:
            text = (text or "").strip()
            if not text or conf < min_conf:
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            rect = QRect(int(min(xs)), int(min(ys)),
                         int(max(xs) - min(xs)), int(max(ys) - min(ys)))
            items.append(TextItem(text, rect, float(conf)))
        return items


# ---------------------------------------------------------------- 내장 엔진: Tesseract
_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


class TesseractEngine(OcrEngine):
    key = "tesseract"
    label = "Tesseract (가벼움·빠름)"

    def _ensure_cmd(self) -> bool:
        import pytesseract
        if shutil.which("tesseract"):
            return True
        for path in _TESSERACT_PATHS:
            if os.path.isfile(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True
        return False

    def is_available(self) -> tuple[bool, str]:
        try:
            import pytesseract
        except Exception as exc:  # noqa: BLE001
            return False, f"pytesseract 모듈이 없습니다: {exc}"
        if not self._ensure_cmd():
            return False, (
                "Tesseract 엔진(tesseract.exe)이 설치되어 있지 않습니다.\n"
                "설치: winget install UB-Mannheim.TesseractOCR\n"
                "(설치 후 다시 선택하세요)"
            )
        try:
            pytesseract.get_tesseract_version()
            return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, f"Tesseract 실행 실패: {exc}"

    def detect(self, base, langs, min_conf):
        import pytesseract
        from pytesseract import Output
        from PIL import Image

        self._ensure_cmd()
        code = "+".join({"en": "eng", "ko": "kor"}.get(x, x) for x in langs)
        pil = Image.fromarray(qimage_to_rgb(base))
        data = pytesseract.image_to_data(
            pil, lang=code, config="--psm 11", output_type=Output.DICT
        )
        items: list[TextItem] = []
        for i in range(len(data["text"])):
            text = (data["text"][i] or "").strip()
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if not text or conf < min_conf * 100:
                continue
            rect = QRect(int(data["left"][i]), int(data["top"][i]),
                         int(data["width"][i]), int(data["height"][i]))
            items.append(TextItem(text, rect, conf / 100.0))
        return items


# ---------------------------------------------------------------- 내장 엔진: RapidOCR
class RapidOcrEngine(OcrEngine):
    key = "rapidocr"
    label = "RapidOCR (ONNX·빠름)"
    priority = -10  # OCR 엔진 목록 최상위(기본 엔진)

    def __init__(self):
        self._engine = None

    def is_available(self) -> tuple[bool, str]:
        try:
            import rapidocr_onnxruntime  # noqa: F401
            return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, (
                "rapidocr_onnxruntime 모듈이 없습니다.\n"
                f"설치: pip install rapidocr_onnxruntime\n({exc})"
            )

    def _get(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
        return self._engine

    def detect(self, base, langs, min_conf):
        res, _elapse = self._get()(qimage_to_rgb(base))
        items: list[TextItem] = []
        if not res:
            return items
        for box, text, score in res:
            text = (text or "").strip()
            conf = float(score)
            if not text or conf < min_conf:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            rect = QRect(int(min(xs)), int(min(ys)),
                         int(max(xs) - min(xs)), int(max(ys) - min(ys)))
            items.append(TextItem(text, rect, conf))
        return items


# ---------------------------------------------------------------- 내장 엔진: Windows OCR
class WindowsOcrEngine(OcrEngine):
    key = "windows"
    label = "Windows OCR (내장·빠름)"

    def is_available(self) -> tuple[bool, str]:
        if sys.platform != "win32":
            return False, "Windows에서만 사용할 수 있습니다."
        try:
            from winrt.windows.media.ocr import OcrEngine as WinOcr
            from winrt.windows.globalization import Language
        except Exception as exc:  # noqa: BLE001
            return False, f"winrt OCR 모듈을 불러올 수 없습니다: {exc}"
        try:
            eng = (WinOcr.try_create_from_user_profile_languages()
                   or WinOcr.try_create_from_language(Language("en")))
        except Exception as exc:  # noqa: BLE001
            return False, f"Windows OCR 초기화 실패: {exc}"
        if eng is None:
            return False, (
                "사용할 수 있는 OCR 언어 팩이 없습니다.\n"
                "설정 → 시간 및 언어 → 언어에서 언어 팩을 추가하세요."
            )
        return True, ""

    def detect(self, base, langs, min_conf):
        png = qimage_to_png_bytes(base)
        # winrt 비동기는 호출 스레드의 COM 아파트먼트(STA)와 충돌해 멈출 수 있으므로
        # MTA인 워커 스레드에서 asyncio 루프를 돌린다.
        result: dict = {}

        def worker():
            import asyncio
            try:
                result["words"] = asyncio.run(_win_recognize(png, tuple(langs)))
            except Exception as exc:  # noqa: BLE001
                result["error"] = exc

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        th.join(timeout=60)
        if th.is_alive():
            raise TimeoutError(
                "Windows OCR이 응답하지 않습니다. 다른 엔진을 사용해 주세요."
            )
        if "error" in result:
            raise result["error"]
        items: list[TextItem] = []
        for text, rect in result.get("words", []):
            text = text.strip()
            if text:
                items.append(TextItem(text, rect, 1.0))  # Windows OCR은 신뢰도 미제공
        return items


async def _win_recognize(png: bytes, langs) -> list[tuple[str, QRect]]:
    from winrt.windows.storage.streams import InMemoryRandomAccessStream, DataWriter
    from winrt.windows.graphics.imaging import BitmapDecoder
    from winrt.windows.media.ocr import OcrEngine as WinOcr
    from winrt.windows.globalization import Language

    stream = InMemoryRandomAccessStream()
    writer = DataWriter(stream)
    writer.write_bytes(png)
    await writer.store_async()
    await writer.flush_async()
    writer.detach_stream()
    stream.seek(0)
    decoder = await BitmapDecoder.create_async(stream)
    bitmap = await decoder.get_software_bitmap_async()

    engine = None
    for tag in list(langs) + ["en"]:
        try:
            engine = WinOcr.try_create_from_language(Language(tag))
        except Exception:  # noqa: BLE001
            engine = None
        if engine:
            break
    if engine is None:
        engine = WinOcr.try_create_from_user_profile_languages()
    if engine is None:
        return []

    result = await engine.recognize_async(bitmap)
    out: list[tuple[str, QRect]] = []
    for line in result.lines:
        for word in line.words:
            r = word.bounding_rect
            out.append((word.text, QRect(int(r.x), int(r.y),
                                         int(r.width), int(r.height))))
    return out


# 하위 호환용 키 상수
ENGINE_EASYOCR = EasyOcrEngine.key
ENGINE_TESSERACT = TesseractEngine.key

# EasyOCR은 사전학습 검출모델(CRAFT)의 상업적 사용 불확실성 때문에 미등록(메뉴 제외).
# 필요 시 아래 주석을 해제하면 다시 사용 가능.
# register_engine(EasyOcrEngine())
register_engine(TesseractEngine())
register_engine(RapidOcrEngine())
register_engine(WindowsOcrEngine())

# 사용자 정의 엔진 자동 로드(있으면). app/custom_ocr.py 에서 register_engine() 호출.
try:
    from . import custom_ocr  # noqa: F401
except Exception:
    pass


# ---------------------------------------------------------------- 마킹 판정
def filter_boxed_text(base: QImage, boxes: list[TextItem],
                      min_sides: int = 3, coverage: float = 0.5) -> list[TextItem]:
    """표제란·표 등 '테두리로 둘러싸인' 글자를 제외한 목록을 반환한다.

    각 텍스트 박스의 상/하/좌/우 바깥쪽 좁은 띠에서 박스 길이의 coverage 이상을
    차지하는 '긴 선'이 있는지 본다. min_sides 면 이상이 선으로 둘러싸이면 표 칸/
    테두리 안(도면 정보)으로 간주해 뺀다. cv2가 없으면 원본을 그대로 반환.
    """
    if not boxes:
        return boxes
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return boxes
    rgb = qimage_to_rgb(base)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    bw = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    hlen = max(15, W // 40)
    vlen = max(15, H // 40)
    horiz = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1)))
    vert = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vlen)))

    def has_h(y0, y1, x0, x1):
        if y1 <= y0 or x1 <= x0:
            return False
        sub = horiz[y0:y1, x0:x1]
        return sub.size > 0 and (sub > 0).sum(axis=1).max() >= coverage * (x1 - x0)

    def has_v(y0, y1, x0, x1):
        if y1 <= y0 or x1 <= x0:
            return False
        sub = vert[y0:y1, x0:x1]
        return sub.size > 0 and (sub > 0).sum(axis=0).max() >= coverage * (y1 - y0)

    kept: list[TextItem] = []
    for it in boxes:
        r = it.rect
        x, y, w, h = r.left(), r.top(), r.width(), r.height()
        x0, x1 = max(0, x), min(W, x + w)
        y0, y1 = max(0, y), min(H, y + h)
        my = max(6, int(1.0 * h))   # 위/아래 탐색 거리
        mx = max(8, int(1.5 * h))   # 좌/우 탐색 거리(글자~칸 테두리)
        sides = 0
        if has_h(max(0, y - my), y, x0, x1):
            sides += 1
        if has_h(y + h, min(H, y + h + my), x0, x1):
            sides += 1
        if has_v(y0, y1, max(0, x - mx), x):
            sides += 1
        if has_v(y0, y1, x + w, min(W, x + w + mx)):
            sides += 1
        if sides < min_sides:
            kept.append(it)
    return kept


def find_unmarked(page: Page, boxes: list[TextItem],
                  alpha_thresh: int = 0) -> list[TextItem]:
    """보이는 주석 레이어 기준으로 마커가 없는 텍스트 항목을 반환한다.

    현재 모드(레이어 그룹)에 속한 레이어만 마킹으로 인정한다(모드별 독립 검사).
    """
    mode = getattr(page, "current_mode", None)
    combined: np.ndarray | None = None
    for layer in page.layers:
        if not layer.visible:
            continue
        if mode is not None and getattr(layer, "group", mode) != mode:
            continue  # 다른 모드 레이어는 검사 대상 아님
        a = _alpha_array(layer.image)
        combined = a if combined is None else np.maximum(combined, a)
    if combined is None:
        return list(boxes)  # 표시된 주석 레이어가 없음 → 전부 미마킹

    h, w = combined.shape
    unmarked: list[TextItem] = []
    for it in boxes:
        r = it.rect
        x0, y0 = max(0, r.left()), max(0, r.top())
        x1, y1 = min(w, r.left() + r.width()), min(h, r.top() + r.height())
        if x1 <= x0 or y1 <= y0:
            unmarked.append(it)
            continue
        if int(combined[y0:y1, x0:x1].max()) <= alpha_thresh:
            unmarked.append(it)
    return unmarked
