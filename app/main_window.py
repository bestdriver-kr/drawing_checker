"""메인 윈도우: 툴바, 레이어 패널, 메뉴, 페이지 이동."""
from __future__ import annotations

import os
import time

from PySide6.QtCore import (
    QEventLoop,
    QPoint,
    QPointF,
    QRectF,
    QSettings,
    QSize,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QToolBar,
)

from .canvas import (
    ERASE_MODE_PIXEL,
    ERASE_MODE_STROKE,
    TOOL_AUTOMARK,
    TOOL_ERASER,
    TOOL_HIGHLIGHTER,
    TOOL_PEN,
    WIDTH_PCT,
    WIDTH_PX,
    Canvas,
)
from .image_loader import load_pages
from .layer import BLEND_MULTIPLY, BLEND_NORMAL, Document
from .ocr import (
    DEVICE_AUTO,
    DEVICE_CPU,
    DEVICE_GPU,
    cuda_available,
    default_engine,
    detect_text_boxes,
    engine_available,
    find_unmarked,
    get_device,
    list_engines,
    set_device,
)
from . import security
from .project import (
    export_ora,
    export_pdf,
    export_tiff,
    flatten_page,
    load_dck,
    save_bundle,
)

APP_VERSION = "1.0"
APP_TITLE = f"Drawing Checker v{APP_VERSION}"

# ---------------------------------------------------------------- UI 언어(i18n)
_UI_LANG = "ko"  # "ko" | "en"


def tr(ko: str, en: str) -> str:
    """현재 UI 언어에 맞는 문자열을 반환."""
    return en if _UI_LANG == "en" else ko


def set_ui_lang(lang: str):
    global _UI_LANG
    _UI_LANG = "en" if lang == "en" else "ko"


def current_ui_lang() -> str:
    return _UI_LANG


def open_filter() -> str:
    return tr(
        "도면/이미지 (*.pdf *.tif *.tiff *.png *.jpg *.jpeg *.bmp);;PDF (*.pdf);;모든 파일 (*.*)",
        "Drawings/Images (*.pdf *.tif *.tiff *.png *.jpg *.jpeg *.bmp);;PDF (*.pdf);;All files (*.*)",
    )


def about_html() -> str:
    if _UI_LANG == "en":
        body = """
<p><b>A program that finds drawing dimensions missing a marker, using OCR.</b></p>
<p>Mark dimensions on drawings/images with pen·highlighter layers, and OCR
highlights dimension items that have no marker.</p>
<b>Features</b>
<ul>
  <li>Open: PDF · multipage TIF · PNG · JPG · BMP</li>
  <li>Tools: pen · highlighter · eraser (whole-stroke / pixel)</li>
  <li>Layers: add·delete·reorder·show/hide·opacity·blend (multiply)</li>
  <li>Mark check (OCR): finds unmarked text (red dashed) / warns on save</li>
  <li>OCR engines: RapidOCR (default) · Drawing OCR · Tesseract · Windows OCR</li>
  <li>Compute device: Auto / GPU / CPU</li>
  <li>Save: <b>.dck</b> (re-editable) + <b>.pdf</b> (toggle layers) together</li>
  <li>Export: PNG · TIFF · PDF (OCG layers) · OpenRaster (.ora)</li>
</ul>
<p style="color:#888;">© 2026 Drawing Checker</p>
"""
    else:
        body = """
<p><b>OCR로 마커가 빠진 도면치수를 찾아주는 프로그램입니다.</b></p>
<p>도면·이미지 위에 펜·형광펜으로 레이어를 덧칠해 검토하고, 마커가 없는
치수 항목을 OCR로 찾아 표시합니다.</p>
<b>주요 기능</b>
<ul>
  <li>열기: PDF · 멀티페이지 TIF · PNG · JPG · BMP</li>
  <li>도구: 펜 · 형광펜 · 지우개(획 전체 / 부분)</li>
  <li>레이어: 추가·삭제·순서·표시숨김·불투명도·블렌드(멀티플라이)</li>
  <li>마킹검사(OCR): 미마킹 텍스트를 빨간 점선으로 표시 / 저장 시 경고</li>
  <li>OCR 엔진: RapidOCR(기본) · Drawing OCR · Tesseract · Windows OCR</li>
  <li>연산 장치: 자동 / GPU / CPU</li>
  <li>저장: <b>.dck</b>(재편집) + <b>.pdf</b>(레이어 토글) 동시</li>
  <li>내보내기: PNG · TIFF · PDF(OCG 레이어) · OpenRaster(.ora)</li>
</ul>
<p style="color:#888;">© 2026 Drawing Checker · 개인/사내 검토용</p>
"""
    return f'<h2>Drawing Checker <span style="color:#888;">v{APP_VERSION}</span></h2>{body}'

# 색상 선택창의 "사용자 지정 색" 기본값 — 자주 쓰는 볼펜색
CUSTOM_PEN_COLORS = ["#000000", "#15268F", "#D81E1E", "#1E8A3C"]  # 검정·파랑·빨강·초록

# 시중에서 흔한 형광펜 색상 (이름, HEX)
PRESET_COLORS = [
    ("형광 노랑", "#FFF200"),
    ("형광 분홍", "#FF4FA3"),
    ("형광 녹색", "#B6FF00"),
    ("형광 파랑", "#00B7FF"),
    ("형광 주황", "#FF9A00"),
    ("형광 코랄", "#FF5A5A"),
    ("형광 민트", "#00E5C0"),
    ("형광 보라", "#B86BFF"),
]


def _make_tool_icon(kind: str) -> QIcon:
    """도구 아이콘을 코드로 그려 QIcon으로 반환(별도 리소스 파일 불필요)."""
    pm = QPixmap(32, 32)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    def round_pen(color: QColor, width: float) -> QPen:
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        return pen

    if kind == TOOL_PEN:
        # 펜: 검은 사선 + 삼각 펜촉
        p.setPen(round_pen(QColor(40, 40, 40), 4))
        p.drawLine(QPointF(9, 23), QPointF(23, 9))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(40, 40, 40))
        p.drawPolygon(QPolygonF([QPointF(6, 26), QPointF(11, 25), QPointF(7, 21)]))
    elif kind == TOOL_HIGHLIGHTER:
        # 형광펜: 굵은 반투명 노란 사선 바
        p.setPen(round_pen(QColor(255, 230, 0, 170), 11))
        p.drawLine(QPointF(8, 24), QPointF(24, 8))
        p.setPen(QPen(QColor(120, 110, 0), 1))
        p.drawLine(QPointF(8, 24), QPointF(24, 8))
    else:  # 지우개
        p.setBrush(QColor(255, 150, 170))
        p.setPen(QPen(QColor(70, 70, 70), 1.5))
        p.translate(16, 16)
        p.rotate(-35)
        p.drawRoundedRect(QRectF(-11, -6, 22, 12), 2, 2)
        p.drawLine(QPointF(2, -6), QPointF(2, 6))
    p.end()
    return QIcon(pm)


def _blueprint_pixmap(s: int) -> QPixmap:
    """도면(블루프린트) 모양 아이콘을 s×s 픽스맵으로 그린다."""
    pm = QPixmap(s, s)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    k = s / 256.0  # 256 기준 좌표 스케일

    def R(x, y, w, h):
        return QRectF(x * k, y * k, w * k, h * k)

    def P(x, y):
        return QPointF(x * k, y * k)

    # 종이(블루프린트) 바탕
    p.setPen(QPen(QColor(150, 195, 245), max(1.0, 2 * k)))
    p.setBrush(QColor(21, 74, 140))
    p.drawRoundedRect(R(20, 20, 216, 216), 16 * k, 16 * k)

    # 격자
    grid = QPen(QColor(255, 255, 255, 38), max(1.0, 1 * k))
    p.setPen(grid)
    for g in range(52, 236, 24):
        p.drawLine(P(g, 28), P(g, 228))
        p.drawLine(P(28, g), P(228, g))

    # 도면 요소(흰 선): 원 + 중심선 + 사각형 + 치수선
    white = QPen(QColor(245, 250, 255), 3.2 * k)
    white.setCapStyle(Qt.PenCapStyle.RoundCap)
    white.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(white)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(P(96, 104), 34 * k, 34 * k)
    center = QPen(QColor(245, 250, 255, 200), 1.6 * k)
    center.setStyle(Qt.PenStyle.DashLine)
    p.setPen(center)
    p.drawLine(P(96, 58), P(96, 150))
    p.drawLine(P(50, 104), P(142, 104))
    p.setPen(white)
    p.drawRect(R(132, 150, 72, 50))

    # 치수선(양끝 화살표)
    dim = QPen(QColor(160, 205, 255), 1.8 * k)
    p.setPen(dim)
    p.drawLine(P(50, 176), P(118, 176))
    p.setBrush(QColor(160, 205, 255))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPolygon(QPolygonF([P(50, 176), P(60, 172), P(60, 180)]))
    p.drawPolygon(QPolygonF([P(118, 176), P(108, 172), P(108, 180)]))

    # 표제란(오른쪽 아래)
    p.setPen(QPen(QColor(245, 250, 255), 2 * k))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(R(150, 206, 70, 22))
    p.drawLine(P(150, 217), P(220, 217))
    p.drawLine(P(185, 206), P(185, 228))
    p.end()
    return pm


def make_app_icon() -> QIcon:
    """여러 해상도를 담은 앱 아이콘."""
    icon = QIcon()
    for size in (256, 128, 64, 48, 32, 16):
        icon.addPixmap(_blueprint_pixmap(size))
    return icon


def _make_action_icon(kind: str) -> QIcon:
    """파일/편집 액션용 아이콘을 코드로 그린다."""
    pm = QPixmap(32, 32)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    def stroke(color: QColor, width: float) -> QPen:
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        return pen

    no_pen = Qt.PenStyle.NoPen

    if kind == "open_image":  # 이미지 열기: 사진(액자 + 해 + 산)
        p.setPen(stroke(QColor(40, 90, 160), 2))
        p.setBrush(QColor(225, 238, 255))
        p.drawRoundedRect(QRectF(4, 7, 24, 18), 3, 3)
        p.setPen(no_pen)
        p.setBrush(QColor(255, 200, 40))
        p.drawEllipse(QPointF(11, 13), 3, 3)
        p.setBrush(QColor(70, 150, 90))
        p.drawPolygon(QPolygonF([QPointF(6, 24), QPointF(14, 15), QPointF(21, 24)]))
        p.setBrush(QColor(95, 175, 115))
        p.drawPolygon(QPolygonF([QPointF(15, 24), QPointF(22, 17), QPointF(27, 24)]))
    elif kind == "open_project":  # 프로젝트 열기: 열린 폴더
        p.setPen(stroke(QColor(150, 110, 20), 1.5))
        p.setBrush(QColor(255, 205, 90))
        p.drawPolygon(QPolygonF([
            QPointF(4, 9), QPointF(12, 9), QPointF(15, 12),
            QPointF(28, 12), QPointF(28, 25), QPointF(4, 25),
        ]))
        p.setBrush(QColor(255, 228, 150))
        p.drawPolygon(QPolygonF([
            QPointF(7, 25), QPointF(11, 15), QPointF(31, 15), QPointF(27, 25),
        ]))
    elif kind == "save":  # 저장: 플로피 디스크
        p.setPen(stroke(QColor(40, 70, 130), 1.5))
        p.setBrush(QColor(70, 120, 200))
        p.drawRoundedRect(QRectF(5, 5, 22, 22), 2, 2)
        p.setPen(no_pen)
        p.setBrush(QColor(225, 235, 250))
        p.drawRect(QRectF(10, 5, 12, 8))
        p.setBrush(QColor(40, 70, 130))
        p.drawRect(QRectF(17, 6, 3, 6))
        p.setBrush(Qt.GlobalColor.white)
        p.drawRect(QRectF(9, 16, 14, 9))
    elif kind == "export_png":  # PNG 내보내기: 사진 + 아래 화살표
        p.setPen(stroke(QColor(90, 90, 90), 2))
        p.setBrush(QColor(240, 240, 240))
        p.drawRoundedRect(QRectF(5, 3, 21, 15), 2, 2)
        p.setPen(no_pen)
        p.setBrush(QColor(255, 200, 40))
        p.drawEllipse(QPointF(11, 8), 2.2, 2.2)
        p.setBrush(QColor(110, 175, 130))
        p.drawPolygon(QPolygonF([QPointF(8, 17), QPointF(15, 10), QPointF(22, 17)]))
        p.setPen(stroke(QColor(40, 140, 70), 2.5))
        p.drawLine(QPointF(16, 19), QPointF(16, 27))
        p.setPen(no_pen)
        p.setBrush(QColor(40, 140, 70))
        p.drawPolygon(QPolygonF([QPointF(11, 25), QPointF(21, 25), QPointF(16, 31)]))
    elif kind in ("undo", "redo"):  # 곡선 화살표
        p.setPen(stroke(QColor(70, 70, 70), 2.6))
        rect = QRectF(7, 9, 18, 15)
        if kind == "undo":
            p.drawArc(rect, 50 * 16, 210 * 16)
            p.setPen(no_pen)
            p.setBrush(QColor(70, 70, 70))
            p.drawPolygon(QPolygonF([QPointF(6, 8), QPointF(15, 9), QPointF(9, 16)]))
        else:
            p.drawArc(rect, 130 * 16, -210 * 16)
            p.setPen(no_pen)
            p.setBrush(QColor(70, 70, 70))
            p.drawPolygon(QPolygonF([QPointF(26, 8), QPointF(17, 9), QPointF(23, 16)]))
    elif kind in ("page_prev", "page_next"):  # 페이지 이동: 굵은 ◀/▶ + 멈춤선
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(40, 110, 200))
        if kind == "page_prev":
            p.drawRect(QRectF(7, 7, 3, 18))  # 끝선
            p.drawPolygon(QPolygonF([QPointF(25, 7), QPointF(25, 25), QPointF(12, 16)]))
        else:
            p.drawRect(QRectF(22, 7, 3, 18))
            p.drawPolygon(QPolygonF([QPointF(7, 7), QPointF(7, 25), QPointF(20, 16)]))
    elif kind == "automark":  # 자동마킹: 노란 박스 + 체크
        p.setPen(stroke(QColor(150, 140, 0), 1.6))
        p.setBrush(QColor(255, 235, 0, 170))
        p.drawRoundedRect(QRectF(5, 9, 22, 14), 2, 2)
        p.setPen(stroke(QColor(40, 140, 70), 3))
        p.drawPolyline(QPolygonF([QPointF(9, 16), QPointF(14, 21), QPointF(24, 7)]))
    elif kind == "next_unmark":  # 다음 미마킹: 빨간 점선 박스 + 오른쪽 화살표
        pen = stroke(QColor(220, 40, 40), 2)
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(5, 9, 13, 14))
        p.setPen(stroke(QColor(40, 110, 200), 2.5))
        p.drawLine(QPointF(20, 16), QPointF(28, 16))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(40, 110, 200))
        p.drawPolygon(QPolygonF([QPointF(24, 11), QPointF(24, 21), QPointF(29, 16)]))
    elif kind == "check":  # 마킹검사: 돋보기 + 체크
        p.setPen(stroke(QColor(60, 110, 60), 2.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(13, 13), 8, 8)
        p.drawLine(QPointF(19, 19), QPointF(27, 27))
        p.setPen(stroke(QColor(40, 160, 70), 2.5))
        p.drawPolyline(QPolygonF([QPointF(9, 13), QPointF(12, 16), QPointF(18, 9)]))
    elif kind == "fit":  # 화면맞춤: 액자 + 대각 확장 화살표
        p.setPen(stroke(QColor(70, 70, 70), 2))
        p.drawRoundedRect(QRectF(5, 5, 22, 22), 2, 2)
        p.drawLine(QPointF(11, 11), QPointF(21, 21))
        p.setPen(no_pen)
        p.setBrush(QColor(70, 70, 70))
        p.drawPolygon(QPolygonF([QPointF(9, 9), QPointF(15, 10), QPointF(10, 15)]))
        p.drawPolygon(QPolygonF([QPointF(23, 23), QPointF(17, 22), QPointF(22, 17)]))
    p.end()
    return QIcon(pm)


def _color_button_style(color: QColor) -> str:
    return (
        f"background-color: {color.name()}; border: 1px solid #888; "
        f"min-width: 35px; min-height: 25px;"
    )


class _OcrWorker(QThread):
    """페이지들의 OCR을 백그라운드에서 수행하고 캐시에 채운다."""

    progressed = Signal(int)      # 완료한 페이지 수
    failed = Signal(str)

    def __init__(self, pages, engine, langs, cache, parent=None):
        super().__init__(parent)
        self._pages = pages
        self._engine = engine
        self._langs = langs
        self._cache = cache
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            for i, page in enumerate(self._pages):
                if self._cancel:
                    return
                key = (id(page), self._engine, self._langs)
                if key not in self._cache:
                    self._cache[key] = detect_text_boxes(
                        page.base, engine=self._engine, langs=self._langs
                    )
                self.progressed.emit(i + 1)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setWindowIcon(make_app_icon())
        self.resize(1280, 860)

        self.canvas = Canvas(self)
        self.setCentralWidget(self.canvas)
        self.setAcceptDrops(True)  # 파일 끌어다 놓기로 열기
        self._layer_controls_updating = False
        self._project_path: str | None = None
        self._ocr_cache: dict = {}  # (id(page), engine, langs) -> list[TextItem]
        self._ocr_engine = default_engine()  # 1순위 엔진(현재 Drawing OCR)
        self._ocr_langs = ("en",)  # OCR 인식 언어(고정: 숫자/영문)
        self._autocheck_on = True  # 저장 시 마킹검사 여부(재빌드에도 유지)
        self._digits_only = False  # 숫자 포함(치수성) 항목만 검사
        self._security_on = False  # 보안(오프라인) 모드
        self._security_warn = False  # True=경고만, False=강제종료
        self._sec_seen = 0  # 경고 모드에서 표시한 위반 수
        self._width_updating = False
        self._nav_unmarked: list = []  # 미마킹 네비게이션 목록(현재 페이지)
        self._nav_idx = -1
        self.canvas.autoMarkRequested.connect(self._on_automark)

        self._build_actions()
        self._build_toolbar()
        self._build_statusbar()

        self.canvas.viewChanged.connect(self._update_status)
        self.canvas.undoStateChanged.connect(self._update_undo_actions)
        self.canvas.layersChanged.connect(self._refresh_layer_combo)
        self.canvas.layersChanged.connect(self._update_progress)
        self.canvas.contentChanged.connect(self._update_progress)
        self._update_undo_actions()
        self._update_status()
        self._refresh_layer_combo()
        self._load_settings()  # 지난 실행의 설정 복원

        self._sec_timer = QTimer(self)  # 경고 모드 위반 폴링
        self._sec_timer.setInterval(1000)
        self._sec_timer.timeout.connect(self._poll_security)
        self._sec_timer.start()

    # ---------- 설정 저장/복원 ----------
    def _settings(self) -> QSettings:
        return QSettings("DrawingChecker", "DrawingChecker")

    def _save_settings(self):
        s = self._settings()
        t = self.canvas.tools
        s.setValue("ui/lang", current_ui_lang())
        s.setValue("ocr/engine", self._ocr_engine)
        s.setValue("ocr/device", get_device())
        s.setValue("ocr/autocheck", self._autocheck_on)
        s.setValue("ocr/digits_only", self._digits_only)
        s.setValue("security/offline", self._security_on)
        s.setValue("security/warn", self._security_warn)
        s.setValue("tool/eraser_mode", t.eraser_mode)
        s.setValue("tool/width_mode", t.width_mode)
        s.setValue("tool/hl_opacity", t.highlighter_opacity)
        s.setValue("color/pen", t.pen_color.name())
        s.setValue("color/hl", t.highlighter_color.name())
        for tool, widths in t.widths.items():
            for unit, val in widths.items():
                s.setValue(f"width/{tool}/{unit}", val)

    def _load_settings(self):
        s = self._settings()
        t = self.canvas.tools
        # 색상
        pen = s.value("color/pen", "")
        if pen:
            t.pen_color = QColor(pen)
            self.pen_color_btn.setStyleSheet(_color_button_style(t.pen_color))
        hl = s.value("color/hl", "")
        if hl:
            t.highlighter_color = QColor(hl)
            self.hl_color_btn.setStyleSheet(_color_button_style(t.highlighter_color))
        t.highlighter_opacity = float(s.value("tool/hl_opacity", t.highlighter_opacity))
        # 굵기(도구·단위별)
        for tool, widths in t.widths.items():
            for unit in list(widths):
                v = s.value(f"width/{tool}/{unit}", None)
                if v is not None:
                    try:
                        widths[unit] = float(v)
                    except (TypeError, ValueError):
                        pass
        # 지우개 방식
        em = s.value("tool/eraser_mode", t.eraser_mode)
        t.eraser_mode = em
        idx = self.eraser_mode_combo.findData(em)
        if idx >= 0:
            self.eraser_mode_combo.setCurrentIndex(idx)
        # 굵기 단위
        wm = s.value("tool/width_mode", t.width_mode)
        t.width_mode = wm
        widx = self.width_unit_combo.findData(wm)
        if widx >= 0:
            self.width_unit_combo.setCurrentIndex(widx)
        self._apply_width_spin_range()
        # OCR 엔진
        eng = s.value("ocr/engine", self._ocr_engine)
        if eng in self._engine_actions and engine_available(eng)[0]:
            self._ocr_engine = eng
            self._engine_actions[eng].setChecked(True)
        # 연산 장치
        dev = s.value("ocr/device", get_device())
        if dev in self._device_actions:
            set_device(dev)
            self._device_actions[dev].setChecked(True)
        # 저장 시 검사
        ac = s.value("ocr/autocheck", True, type=bool)
        self._autocheck_on = ac
        self.act_autocheck.setChecked(ac)
        # 숫자·치수만 검사
        do = s.value("ocr/digits_only", False, type=bool)
        self._digits_only = do
        self.act_digits_only.setChecked(do)
        # 보안 모드
        self._security_on = s.value("security/offline", False, type=bool)
        self._security_warn = s.value("security/warn", False, type=bool)
        self._apply_security_state()
        # UI 언어(필요 시 즉시 재번역)
        lang = s.value("ui/lang", current_ui_lang())
        if lang in ("ko", "en") and lang != current_ui_lang():
            self._set_ui_lang(lang)

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    # ---------- 드래그앤드롭 열기 ----------
    _DROP_EXTS = (".dck", ".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

    def _first_supported_drop(self, event):
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if p and os.path.splitext(p)[1].lower() in self._DROP_EXTS:
                return p
        return None

    def dragEnterEvent(self, event):
        if self._first_supported_drop(event):
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = self._first_supported_drop(event)
        if path:
            event.acceptProposedAction()
            self._open_path(path)

    def _open_path(self, path: str):
        """확장자에 따라 프로젝트(.dck) 또는 이미지/PDF로 연다."""
        if os.path.splitext(path)[1].lower() == ".dck":
            try:
                doc = load_dck(path)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "열기 실패", f"프로젝트를 열 수 없습니다:\n{exc}")
                return
            self._project_path = path
        else:
            try:
                pages = load_pages(path)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "열기 실패", f"이미지를 열 수 없습니다:\n{exc}")
                return
            doc = Document(pages, path)
            self._project_path = None
        self._ocr_cache.clear()
        self._nav_unmarked = []
        self._nav_idx = -1
        self.canvas.set_document(doc)
        self.setWindowTitle(f"{APP_TITLE} — {os.path.basename(path)}")
        self._update_status()

    # ---------- 액션/메뉴 ----------
    def _build_actions(self):
        self.act_open = QAction(_make_action_icon("open_image"), tr("이미지 열기", "Open Image"), self)
        self.act_open.setIconText(tr("이미지", "Image"))
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.setToolTip(tr("이미지 열기 (Ctrl+O)", "Open image (Ctrl+O)"))
        self.act_open.triggered.connect(self.open_file)

        self.act_open_project = QAction(_make_action_icon("open_project"), tr("프로젝트 열기", "Open Project"), self)
        self.act_open_project.setIconText(tr("프로젝트", "Project"))
        self.act_open_project.setShortcut("Ctrl+Shift+O")
        self.act_open_project.setToolTip(tr("프로젝트(.dck) 열기 (Ctrl+Shift+O)", "Open project (.dck) (Ctrl+Shift+O)"))
        self.act_open_project.triggered.connect(self.open_project)

        self.act_save_project = QAction(_make_action_icon("save"), tr("프로젝트 저장(.dck + .pdf)", "Save Project (.dck + .pdf)"), self)
        self.act_save_project.setIconText(tr("저장", "Save"))
        self.act_save_project.setShortcut(QKeySequence.StandardKey.Save)
        self.act_save_project.setToolTip(tr("프로젝트 저장: .dck + .pdf 함께 (Ctrl+S)", "Save project: .dck + .pdf (Ctrl+S)"))
        self.act_save_project.triggered.connect(lambda: self.save_project(False))

        self.act_save_project_as = QAction(tr("다른 이름으로 저장(.dck + .pdf)", "Save As (.dck + .pdf)"), self)
        self.act_save_project_as.setShortcut("Ctrl+Shift+S")
        self.act_save_project_as.triggered.connect(lambda: self.save_project(True))

        self.act_export = QAction(_make_action_icon("export_png"), tr("PNG 내보내기", "Export PNG"), self)
        self.act_export.setIconText("PNG")
        self.act_export.setShortcut("Ctrl+E")
        self.act_export.setToolTip(tr("PNG로 내보내기 (Ctrl+E)", "Export to PNG (Ctrl+E)"))
        self.act_export.triggered.connect(self.export_png)

        self.act_export_ora = QAction(tr("OpenRaster(.ora) 내보내기", "Export OpenRaster (.ora)"), self)
        self.act_export_ora.triggered.connect(self.export_ora_current)

        self.act_export_tiff = QAction(tr("TIFF 내보내기", "Export TIFF"), self)
        self.act_export_tiff.triggered.connect(self.export_tiff_doc)

        self.act_export_pdf = QAction(tr("PDF 내보내기(레이어 토글)", "Export PDF (toggle layers)"), self)
        self.act_export_pdf.triggered.connect(self.export_pdf_doc)

        self.act_exit = QAction(tr("종료", "Exit"), self)
        self.act_exit.setShortcut("Ctrl+Q")
        self.act_exit.setToolTip(tr("프로그램 종료 (Ctrl+Q)", "Quit the program (Ctrl+Q)"))
        self.act_exit.triggered.connect(self.close)

        self.act_undo = QAction(_make_action_icon("undo"), tr("실행취소", "Undo"), self)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_undo.setToolTip(tr("실행취소 (Ctrl+Z)", "Undo (Ctrl+Z)"))
        self.act_undo.triggered.connect(self.canvas.undo)

        self.act_redo = QAction(_make_action_icon("redo"), tr("다시실행", "Redo"), self)
        self.act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.act_redo.setToolTip(tr("다시실행 (Ctrl+Y)", "Redo (Ctrl+Y)"))
        self.act_redo.triggered.connect(self.canvas.redo)

        self.act_fit = QAction(_make_action_icon("fit"), tr("화면맞춤", "Fit"), self)
        self.act_fit.setShortcut("Ctrl+0")
        self.act_fit.setToolTip(tr("화면맞춤 (Ctrl+0)", "Fit to window (Ctrl+0)"))
        self.act_fit.triggered.connect(self.canvas.fit_to_window)

        self.act_check = QAction(_make_action_icon("check"), tr("마킹검사", "Mark Check"), self)
        self.act_check.setIconText(tr("마킹검사", "Check"))
        self.act_check.setToolTip(tr("OCR로 텍스트를 찾아 마커가 없는 항목을 표시", "Find text via OCR and show unmarked items"))
        self.act_check.triggered.connect(self.check_marks)

        self.act_next_unmark = QAction(_make_action_icon("next_unmark"), tr("다음 미마킹", "Next Unmarked"), self)
        self.act_next_unmark.setIconText(tr("다음", "Next"))
        self.act_next_unmark.setShortcut("N")
        self.act_next_unmark.setToolTip(tr("다음 미마킹 항목으로 이동 (N)", "Go to next unmarked item (N)"))
        self.act_next_unmark.triggered.connect(self.goto_next_unmarked)

        self.act_autocheck = QAction(tr("저장 시 마킹검사", "Mark check on save"), self)
        self.act_autocheck.setCheckable(True)
        self.act_autocheck.setChecked(self._autocheck_on)
        self.act_autocheck.setToolTip(tr("저장/내보내기 전에 미마킹 항목을 자동 검사", "Auto-check unmarked items before save/export"))
        self.act_autocheck.toggled.connect(lambda v: setattr(self, "_autocheck_on", v))

        self.act_digits_only = QAction(tr("숫자·치수 항목만 검사", "Check numeric items only"), self)
        self.act_digits_only.setCheckable(True)
        self.act_digits_only.setChecked(self._digits_only)
        self.act_digits_only.setToolTip(
            tr("숫자가 포함된 항목(치수/공차)만 검사하고 일반 글자는 제외",
               "Only check items containing digits (dimensions/tolerances); skip plain words"))
        self.act_digits_only.toggled.connect(self._on_toggle_digits_only)

        menu = self.menuBar()
        file_menu = menu.addMenu(tr("파일", "File"))
        file_menu.addAction(self.act_open)
        file_menu.addAction(self.act_open_project)
        file_menu.addSeparator()
        file_menu.addAction(self.act_save_project)
        file_menu.addAction(self.act_save_project_as)
        file_menu.addSeparator()
        export_menu = file_menu.addMenu(tr("내보내기", "Export"))
        export_menu.addAction(self.act_export)
        export_menu.addAction(self.act_export_tiff)
        export_menu.addAction(self.act_export_pdf)
        export_menu.addAction(self.act_export_ora)
        file_menu.addSeparator()
        file_menu.addAction(self.act_exit)
        edit_menu = menu.addMenu(tr("편집", "Edit"))
        edit_menu.addAction(self.act_undo)
        edit_menu.addAction(self.act_redo)
        view_menu = menu.addMenu(tr("보기", "View"))
        view_menu.addAction(self.act_fit)
        check_menu = menu.addMenu(tr("검사", "Check"))
        check_menu.addAction(self.act_check)
        check_menu.addAction(self.act_next_unmark)
        check_menu.addAction(self.act_autocheck)
        check_menu.addAction(self.act_digits_only)
        engine_menu = check_menu.addMenu(tr("OCR 엔진", "OCR Engine"))
        self.engine_group = QActionGroup(self)
        self.engine_group.setExclusive(True)
        self._engine_actions: dict[str, QAction] = {}
        default_key = default_engine()
        for key, label in list_engines():
            if key == default_key:
                label = label + tr(" (기본)", " (default)")
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(key == self._ocr_engine)
            act.triggered.connect(lambda _c=False, k=key: self._set_ocr_engine(k))
            self.engine_group.addAction(act)
            engine_menu.addAction(act)
            self._engine_actions[key] = act

        # 연산 장치 선택 (Drawing OCR 등 torch 엔진에 적용)
        device_menu = check_menu.addMenu(tr("연산 장치", "Compute Device"))
        self.device_group = QActionGroup(self)
        self.device_group.setExclusive(True)
        self._device_actions: dict[str, QAction] = {}
        gpu_ok = cuda_available()
        na = tr(" · 사용 불가", " · unavailable")
        for key, label in (
            (DEVICE_AUTO, tr("자동 (CUDA 있으면 GPU)", "Auto (GPU if CUDA)")),
            (DEVICE_GPU, "GPU (CUDA)" + ("" if gpu_ok else na)),
            (DEVICE_CPU, "CPU"),
        ):
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(key == get_device())
            act.triggered.connect(lambda _c=False, k=key: self._set_ocr_device(k))
            self.device_group.addAction(act)
            device_menu.addAction(act)
            self._device_actions[key] = act

        # UI 언어
        lang_menu = menu.addMenu(tr("언어", "Language"))
        self.uilang_group = QActionGroup(self)
        self.uilang_group.setExclusive(True)
        self._uilang_actions: dict[str, QAction] = {}
        for code, label in (("ko", "한국어"), ("en", "English")):
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(code == current_ui_lang())
            act.triggered.connect(lambda _c=False, lc=code: self._set_ui_lang(lc))
            self.uilang_group.addAction(act)
            lang_menu.addAction(act)
            self._uilang_actions[code] = act

        sec_menu = menu.addMenu(tr("보안", "Security"))
        self.act_security = QAction(
            tr("보안 모드 (외부 통신 차단·유출 시 강제종료)",
               "Security mode (block network · force-quit on leak)"), self)
        self.act_security.setCheckable(True)
        self.act_security.setChecked(self._security_on)  # 연결 전 설정 → 미발동
        self.act_security.toggled.connect(self._on_toggle_security)
        sec_menu.addAction(self.act_security)

        self.act_security_warn = QAction(
            tr("강제종료 대신 경고만", "Warn only (don't force-quit)"), self)
        self.act_security_warn.setCheckable(True)
        self.act_security_warn.setChecked(self._security_warn)
        self.act_security_warn.setToolTip(
            tr("경고만: 외부 연결을 막되 종료하지 않고 상태바에 경고 표시",
               "Warn only: block external connections without quitting; show a warning"))
        self.act_security_warn.toggled.connect(self._on_toggle_security_warn)
        sec_menu.addAction(self.act_security_warn)

        help_menu = menu.addMenu(tr("도움말", "Help"))
        self.act_about = QAction(tr("프로그램 정보", "About"), self)
        self.act_about.triggered.connect(self.show_about)
        help_menu.addAction(self.act_about)

    def show_about(self):
        QMessageBox.about(self, tr("Drawing Checker 정보", "About Drawing Checker"),
                          about_html())

    def _build_toolbar(self):
        tb = QToolBar("도구")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.setIconSize(QSize(35, 35))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        tb.addAction(self.act_open)
        tb.addAction(self.act_open_project)
        tb.addAction(self.act_save_project)
        tb.addAction(self.act_fit)
        tb.addAction(self.act_check)
        tb.addAction(self.act_next_unmark)
        tb.addSeparator()
        tb.addAction(self.act_undo)
        tb.addAction(self.act_redo)

        # 페이지 이동
        tb.addSeparator()
        tb.addWidget(QLabel(tr(" 페이지 ", " Page ")))
        self.prev_btn = QPushButton(_make_action_icon("page_prev"), tr(" 이전", " Prev"))
        self.prev_btn.setToolTip(tr("이전 페이지", "Previous page"))
        self.prev_btn.clicked.connect(lambda: self._change_page(-1))
        tb.addWidget(self.prev_btn)

        self.page_nav_label = QLabel("– / –")
        self.page_nav_label.setStyleSheet(
            "font-weight: bold; padding: 0 6px; min-width: 44px;"
        )
        self.page_nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tb.addWidget(self.page_nav_label)

        self.next_btn = QPushButton(_make_action_icon("page_next"), tr(" 다음", " Next"))
        self.next_btn.setToolTip(tr("다음 페이지", "Next page"))
        self.next_btn.clicked.connect(lambda: self._change_page(+1))
        tb.addWidget(self.next_btn)

        # 두 번째 줄: 레이어 + 도구
        self.addToolBarBreak()
        pb = QToolBar("layer-tool")
        pb.setMovable(False)
        self.addToolBar(pb)

        # 레이어 그룹 (지정 색상 바로 왼쪽)
        pb.addWidget(QLabel(tr(" 레이어 ", " Layer ")))
        self.layer_combo = QComboBox()
        self.layer_combo.setMinimumWidth(188)
        self.layer_combo.setToolTip(tr("활성 레이어 선택 / 맨 아래 항목으로 새 레이어 추가",
                                       "Select active layer / last item adds a new layer"))
        self.layer_combo.currentIndexChanged.connect(self._on_layer_combo_changed)
        pb.addWidget(self.layer_combo)

        self.layer_rename_btn = QPushButton("✎")
        self.layer_rename_btn.setToolTip(tr("활성 레이어 이름 변경", "Rename active layer"))
        self.layer_rename_btn.setFixedWidth(35)
        self.layer_rename_btn.clicked.connect(self._rename_active_layer)
        pb.addWidget(self.layer_rename_btn)

        self.layer_delete_btn = QPushButton("✕")
        self.layer_delete_btn.setToolTip(tr("활성 레이어 삭제", "Delete active layer"))
        self.layer_delete_btn.setFixedWidth(35)
        self.layer_delete_btn.clicked.connect(self._delete_active_layer)
        pb.addWidget(self.layer_delete_btn)

        self.layer_visible_btn = QPushButton("👁")
        self.layer_visible_btn.setCheckable(True)
        self.layer_visible_btn.setToolTip(tr("활성 레이어 표시/숨김", "Show/hide active layer"))
        self.layer_visible_btn.setFixedWidth(35)
        self.layer_visible_btn.toggled.connect(self._on_visible_toggled)
        pb.addWidget(self.layer_visible_btn)

        self.layer_up_btn = QPushButton("▲")
        self.layer_up_btn.setToolTip(tr("레이어 위로", "Move layer up"))
        self.layer_up_btn.setFixedWidth(30)
        self.layer_up_btn.clicked.connect(lambda: self._move_active_layer(+1))
        pb.addWidget(self.layer_up_btn)

        self.layer_down_btn = QPushButton("▼")
        self.layer_down_btn.setToolTip(tr("레이어 아래로", "Move layer down"))
        self.layer_down_btn.setFixedWidth(30)
        self.layer_down_btn.clicked.connect(lambda: self._move_active_layer(-1))
        pb.addWidget(self.layer_down_btn)

        # 레이어 오른쪽: 도구(작은 아이콘) + 펜색/형광색/굵기/지우개 방식
        pb.addSeparator()
        pb.setIconSize(QSize(24, 24))
        pb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_actions: dict[str, QAction] = {}
        for tool, label, shortcut in (
            (TOOL_PEN, tr("펜", "Pen"), "P"),
            (TOOL_HIGHLIGHTER, tr("형광펜", "Highlighter"), "H"),
            (TOOL_ERASER, tr("지우개", "Eraser"), "E"),
            (TOOL_AUTOMARK, tr("자동마킹", "Auto-mark"), "A"),
        ):
            icon = (_make_action_icon("automark") if tool == TOOL_AUTOMARK
                    else _make_tool_icon(tool))
            act = QAction(icon, f"{label}({shortcut})", self)
            act.setCheckable(True)
            act.setShortcut(shortcut)
            act.setToolTip(f"{label} ({shortcut})")
            act.triggered.connect(lambda _checked=False, t=tool: self._select_tool(t))
            self.tool_group.addAction(act)
            pb.addAction(act)
            self.tool_actions[tool] = act
        self.tool_actions[self.canvas.tools.tool].setChecked(True)

        self.pen_color_btn = QPushButton()
        self.pen_color_btn.setToolTip(tr("펜 색상", "Pen color"))
        self.pen_color_btn.setStyleSheet(_color_button_style(self.canvas.tools.pen_color))
        self.pen_color_btn.clicked.connect(self._pick_pen_color)
        pb.addWidget(QLabel(tr(" 펜 ", " Pen ")))
        pb.addWidget(self.pen_color_btn)

        self.hl_color_btn = QPushButton()
        self.hl_color_btn.setToolTip(tr("형광펜 색상", "Highlighter color"))
        self.hl_color_btn.setStyleSheet(
            _color_button_style(self.canvas.tools.highlighter_color)
        )
        self.hl_color_btn.clicked.connect(self._pick_hl_color)
        pb.addWidget(QLabel(tr(" 형광 ", " HL ")))
        pb.addWidget(self.hl_color_btn)

        pb.addWidget(QLabel(tr(" 굵기 ", " Width ")))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.valueChanged.connect(self._on_width_changed)
        pb.addWidget(self.width_spin)

        self.width_unit_combo = QComboBox()
        self.width_unit_combo.addItem("px", WIDTH_PX)
        self.width_unit_combo.addItem(tr("이미지%", "image%"), WIDTH_PCT)
        self.width_unit_combo.setToolTip(tr(
            "px: 이미지 픽셀 절대값 / 이미지%: 이미지 폭 대비 비율(해상도 무관)",
            "px: absolute image pixels / image%: ratio of image width (resolution-independent)"))
        # 현재 기본 단위에 콤보 선택을 맞춤(연결 전 설정 → 핸들러 미발동)
        self.width_unit_combo.setCurrentIndex(
            self.width_unit_combo.findData(self.canvas.tools.width_mode)
        )
        self.width_unit_combo.currentIndexChanged.connect(self._on_width_unit_changed)
        pb.addWidget(self.width_unit_combo)
        self._apply_width_spin_range()  # 초기 범위/값 설정

        self.eraser_mode_combo = QComboBox()
        self.eraser_mode_combo.addItem(tr("획 전체", "Whole stroke"), ERASE_MODE_STROKE)
        self.eraser_mode_combo.addItem(tr("부분(픽셀)", "Pixel"), ERASE_MODE_PIXEL)
        self.eraser_mode_combo.setToolTip(tr(
            "획 전체: 클릭한 획을 통째로 삭제 / 부분(픽셀): 칠한 부분만 지움",
            "Whole stroke: delete the clicked stroke / Pixel: erase only painted area"))
        idx = self.eraser_mode_combo.findData(self.canvas.tools.eraser_mode)
        if idx >= 0:
            self.eraser_mode_combo.setCurrentIndex(idx)
        self.eraser_mode_combo.currentIndexChanged.connect(self._on_eraser_mode_changed)
        pb.addWidget(QLabel(tr(" 지우개 ", " Eraser ")))
        pb.addWidget(self.eraser_mode_combo)

        # 세 번째 줄: 지정 색상 + 불투명도 / 블렌드
        self.addToolBarBreak()
        cb = QToolBar("color-opacity")
        cb.setMovable(False)
        self.addToolBar(cb)
        cb.addWidget(QLabel(tr(" 지정 색상 ", " Colors ")))
        cha = tr("차", "")
        for i, (name, hex_code) in enumerate(PRESET_COLORS):
            color = QColor(hex_code)
            swatch = QPushButton()
            swatch.setToolTip(f"{i + 1}{cha} ({hex_code})")
            swatch.setFixedSize(33, 28)
            swatch.setStyleSheet(_color_button_style(color))
            swatch.clicked.connect(lambda _checked=False, c=color: self._apply_preset(c))
            cb.addWidget(swatch)

        cb.addSeparator()
        cb.addWidget(QLabel(tr(" 불투명도 ", " Opacity ")))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setFixedWidth(150)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        cb.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("100%")
        self.opacity_label.setFixedWidth(50)
        cb.addWidget(self.opacity_label)

        cb.addWidget(QLabel(tr(" 블렌드 ", " Blend ")))
        self.blend_combo = QComboBox()
        self.blend_combo.addItem(tr("노멀", "Normal"), BLEND_NORMAL)
        self.blend_combo.addItem(tr("멀티플라이(형광펜용)", "Multiply (for highlighter)"), BLEND_MULTIPLY)
        idx = self.blend_combo.findData(self.canvas.page.active_layer.blend) if self.canvas.page else -1
        if idx >= 0:
            self.blend_combo.setCurrentIndex(idx)
        self.blend_combo.currentIndexChanged.connect(self._on_blend_changed)
        cb.addWidget(self.blend_combo)

        self._toolbars = [tb, pb, cb]  # 언어 전환 시 재생성용

    def _apply_preset(self, color: QColor):
        """프리셋 색을 현재 선택된 도구(펜이면 펜색, 그 외엔 형광색)에 적용."""
        if self.canvas.tools.tool == TOOL_PEN:
            self.canvas.tools.pen_color = QColor(color)
            self.pen_color_btn.setStyleSheet(_color_button_style(color))
        else:
            self.canvas.tools.highlighter_color = QColor(color)
            self.hl_color_btn.setStyleSheet(_color_button_style(color))

    def _build_statusbar(self):
        self.status = self.statusBar()
        self.zoom_label = QLabel("")
        self.page_label = QLabel("")
        self.security_label = QLabel("")
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("font-weight: bold; padding: 0 8px;")
        self.version_label = QLabel(f"v{APP_VERSION}")
        self.version_label.setStyleSheet("color:#888; padding:0 8px;")
        self.status.addPermanentWidget(self.security_label)
        self.status.addPermanentWidget(self.progress_label)
        self.status.addPermanentWidget(self.page_label)
        self.status.addPermanentWidget(self.zoom_label)
        self.status.addPermanentWidget(self.version_label)

    # ---------- 핸들러 ----------
    # ---------- 레이어 드롭다운 ----------
    ADD_LAYER_DATA = -1

    def _refresh_layer_combo(self):
        """레이어 드롭다운과 불투명도·블렌드·표시·순서 컨트롤을 활성 레이어에 맞춰 동기화."""
        page = self.canvas.page
        has_page = page is not None
        self._layer_controls_updating = True

        self.layer_combo.clear()
        if has_page:
            # 상단 레이어가 위에 오도록 역순으로 표시
            for index in range(len(page.layers) - 1, -1, -1):
                self.layer_combo.addItem(page.layers[index].name, index)
            self.layer_combo.addItem("➕ 새 레이어 추가", self.ADD_LAYER_DATA)
            active_row = len(page.layers) - 1 - page.active_index
            self.layer_combo.setCurrentIndex(active_row)

            active = page.active_layer
            self.opacity_slider.setValue(int(active.opacity * 100))
            self.opacity_label.setText(f"{int(active.opacity * 100)}%")
            self.blend_combo.setCurrentIndex(self.blend_combo.findData(active.blend))
            self.layer_visible_btn.setChecked(active.visible)

        self.layer_combo.setEnabled(has_page)
        self.layer_rename_btn.setEnabled(has_page)
        self.layer_visible_btn.setEnabled(has_page)
        self.opacity_slider.setEnabled(has_page)
        self.blend_combo.setEnabled(has_page)
        # 마지막 한 장은 삭제 불가
        self.layer_delete_btn.setEnabled(has_page and len(page.layers) > 1)
        self.layer_up_btn.setEnabled(has_page and page.active_index < len(page.layers) - 1)
        self.layer_down_btn.setEnabled(has_page and page.active_index > 0)
        self._layer_controls_updating = False

    def _on_layer_combo_changed(self, _index: int):
        if self._layer_controls_updating:
            return
        page = self.canvas.page
        if page is None:
            return
        data = self.layer_combo.currentData()
        if data == self.ADD_LAYER_DATA:
            page.add_layer()  # 활성 레이어 위에 추가하고 활성화
        else:
            page.active_index = data
        self.canvas.notify_layers_changed()  # 캔버스/컨트롤 모두 갱신

    def _on_visible_toggled(self, checked: bool):
        if self._layer_controls_updating or self.canvas.page is None:
            return
        self.canvas.page.active_layer.visible = checked
        self.canvas.notify_layers_changed()

    def _move_active_layer(self, delta: int):
        if self.canvas.page is None:
            return
        if self.canvas.page.move_active(delta):
            self.canvas.notify_layers_changed()

    def _on_opacity_changed(self, value: int):
        if self._layer_controls_updating or self.canvas.page is None:
            return
        self.canvas.page.active_layer.opacity = value / 100.0
        self.opacity_label.setText(f"{value}%")
        self.canvas.notify_layers_changed()

    def _on_blend_changed(self, _index: int):
        if self._layer_controls_updating or self.canvas.page is None:
            return
        self.canvas.page.active_layer.blend = self.blend_combo.currentData()
        self.canvas.notify_layers_changed()

    def _rename_active_layer(self):
        page = self.canvas.page
        if page is None:
            return
        layer = page.active_layer
        new_name, ok = QInputDialog.getText(
            self, tr("레이어 이름 변경", "Rename layer"),
            tr("새 이름:", "New name:"), text=layer.name
        )
        if ok and new_name.strip():
            layer.name = new_name.strip()
            self.canvas.notify_layers_changed()

    def _delete_active_layer(self):
        page = self.canvas.page
        if page is None:
            return
        if not page.remove_active_layer():
            QMessageBox.information(self, tr("삭제 불가", "Cannot delete"),
                                    tr("마지막 레이어는 삭제할 수 없습니다.",
                                       "The last layer cannot be deleted."))
            return
        self.canvas.notify_layers_changed()

    def _current_tool_width(self) -> float:
        return self.canvas.tools.get_width(self.canvas.tools.tool)

    def _apply_width_spin_range(self):
        """현재 굵기 단위에 맞춰 스핀 범위/표시값을 설정."""
        self._width_updating = True
        if self.canvas.tools.width_mode == WIDTH_PCT:
            self.width_spin.setDecimals(2)
            self.width_spin.setRange(0.05, 50.0)
            self.width_spin.setSingleStep(0.1)
        else:
            self.width_spin.setDecimals(1)
            self.width_spin.setRange(0.5, 500.0)
            self.width_spin.setSingleStep(1.0)
        self.width_spin.setValue(self._current_tool_width())
        self._width_updating = False

    def _select_tool(self, tool: str):
        self.canvas.tools.tool = tool
        if tool in self.tool_actions:
            self.tool_actions[tool].setChecked(True)
        # 굵기 스핀을 도구별 값으로 동기화
        self._width_updating = True
        self.width_spin.setValue(self._current_tool_width())
        self._width_updating = False

    def _on_width_changed(self, value: float):
        if self._width_updating:
            return
        self.canvas.tools.set_width(self.canvas.tools.tool, value)

    def _on_width_unit_changed(self, _index: int):
        # px 값과 % 값을 각각 따로 기억 → 전환해도 서로 영향 없음(환산하지 않음)
        new_mode = self.width_unit_combo.currentData()
        if new_mode == self.canvas.tools.width_mode:
            return
        self.canvas.tools.width_mode = new_mode
        self._apply_width_spin_range()
        unit = "이미지%" if new_mode == WIDTH_PCT else "px"
        self.status.showMessage(f"굵기 단위: {unit}", 3000)

    def _on_eraser_mode_changed(self, _index: int):
        self.canvas.tools.eraser_mode = self.eraser_mode_combo.currentData()

    def _set_custom_colors(self, hex_list):
        """색상 선택창의 '사용자 지정 색' 슬롯을 왼쪽 위부터 채운다."""
        for i, hex_code in enumerate(hex_list):
            c = QColor(hex_code)
            try:
                QColorDialog.setCustomColor(i, c)
            except TypeError:
                QColorDialog.setCustomColor(i, c.rgb())

    def _pick_pen_color(self):
        self._set_custom_colors(CUSTOM_PEN_COLORS)  # 볼펜색 4개(왼쪽 위부터)
        color = QColorDialog.getColor(self.canvas.tools.pen_color, self,
                                      tr("펜 색상", "Pen color"))
        if color.isValid():
            self.canvas.tools.pen_color = color
            self.pen_color_btn.setStyleSheet(_color_button_style(color))

    def _pick_hl_color(self):
        # 형광펜: 툴바 프리셋 1~8을 사용자 지정 색에 왼쪽 위→오른쪽으로 채움
        self._set_custom_colors([hexc for _name, hexc in PRESET_COLORS])
        color = QColorDialog.getColor(self.canvas.tools.highlighter_color, self,
                                      tr("형광펜 색상", "Highlighter color"))
        if color.isValid():
            self.canvas.tools.highlighter_color = color
            self.hl_color_btn.setStyleSheet(_color_button_style(color))

    def _change_page(self, delta: int):
        doc = self.canvas.doc
        if doc is None:
            return
        if doc.set_page(doc.current_index + delta):
            self.canvas.clear_overlay()
            self._nav_unmarked = []
            self._nav_idx = -1
            self.canvas.notify_layers_changed()
            self.canvas._rebuild_caches()
            self.canvas.fit_to_window()
            self._update_status()

    def _update_undo_actions(self):
        self.act_undo.setEnabled(self.canvas.can_undo())
        self.act_redo.setEnabled(self.canvas.can_redo())

    def _update_status(self):
        doc = self.canvas.doc
        if doc is None:
            self.zoom_label.setText("")
            self.page_label.setText(tr("파일 없음", "No file"))
            self.page_nav_label.setText("– / –")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self._update_progress()
            return
        self.zoom_label.setText(tr("줌 ", "Zoom ") + f"{self.canvas.scale * 100:.0f}%")
        self.page_label.setText(
            tr("페이지 ", "Page ") + f"{doc.current_index + 1}/{doc.page_count}"
        )
        self.page_nav_label.setText(f"{doc.current_index + 1} / {doc.page_count}")
        self.prev_btn.setEnabled(doc.current_index > 0)
        self.next_btn.setEnabled(doc.current_index < doc.page_count - 1)
        self._update_progress()

    def _update_progress(self):
        """현재 페이지의 마킹 진척도를 상태바에 표시(검사 후에만)."""
        page = self.canvas.page
        if page is None:
            self.progress_label.setText("")
            return
        boxes = self._get_boxes(page)
        if not boxes:
            self.progress_label.setText(tr("마킹: 검사 전", "Mark: not checked"))
            return
        total = len(boxes)
        unmarked = len(find_unmarked(page, boxes))
        marked = total - unmarked
        pct = int(round(marked / total * 100)) if total else 0
        self.progress_label.setText(tr(
            f"마킹 {marked}/{total} ({pct}%) · 남음 {unmarked}",
            f"Marked {marked}/{total} ({pct}%) · left {unmarked}"))

    # ---------- 파일 ----------
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("이미지 열기", "Open Image"), "", open_filter())
        if not path:
            return
        try:
            pages = load_pages(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, tr("열기 실패", "Open failed"),
                                 tr("이미지를 열 수 없습니다:\n", "Cannot open image:\n") + str(exc))
            return
        doc = Document(pages, path)
        self._project_path = None  # 새 이미지는 아직 프로젝트로 저장된 적 없음
        self._ocr_cache.clear()
        self._nav_unmarked = []
        self._nav_idx = -1
        self.canvas.set_document(doc)
        self.setWindowTitle(f"{APP_TITLE} — {os.path.basename(path)}")
        self._update_status()

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "프로젝트 열기", "", "DCK 프로젝트 (*.dck)"
        )
        if not path:
            return
        try:
            doc = load_dck(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "열기 실패", f"프로젝트를 열 수 없습니다:\n{exc}")
            return
        self._project_path = path
        self._ocr_cache.clear()
        self._nav_unmarked = []
        self._nav_idx = -1
        self.canvas.set_document(doc)
        self.setWindowTitle(f"{APP_TITLE} — {os.path.basename(path)}")
        self._update_status()

    def save_project(self, save_as: bool):
        if self.canvas.doc is None:
            QMessageBox.information(self, "저장", "먼저 이미지나 프로젝트를 여세요.")
            return
        if not self._confirm_before_save():
            return
        path = self._project_path
        if save_as or not path:
            default = "프로젝트.dck"
            src = self.canvas.doc.path
            if src:
                default = os.path.splitext(os.path.basename(src))[0] + ".dck"
            path, _ = QFileDialog.getSaveFileName(
                self, "프로젝트 저장", default, "DCK 프로젝트 (*.dck)"
            )
            if not path:
                return
            if not path.lower().endswith(".dck"):
                path += ".dck"
        try:
            written = save_bundle(self.canvas.doc, path)  # .dck + .ora 함께 저장
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "저장 실패", f"프로젝트를 저장할 수 없습니다:\n{exc}")
            return
        self._project_path = path
        self.canvas.doc.path = path
        self.setWindowTitle(f"{APP_TITLE} — {os.path.basename(path)}")
        names = ", ".join(os.path.basename(w) for w in written)
        self.status.showMessage(f"저장됨({len(written)}개): {names}", 5000)

    def export_ora_current(self):
        page = self.canvas.page
        if page is None:
            QMessageBox.information(self, "내보내기", "먼저 이미지를 여세요.")
            return
        if not self._confirm_before_save():
            return
        merged = self.canvas.render_flat()
        doc = self.canvas.doc
        default = "export.ora"
        if doc and doc.path:
            stem = os.path.splitext(os.path.basename(doc.path))[0]
            suffix = f"_p{doc.current_index + 1}" if doc.page_count > 1 else ""
            default = f"{stem}{suffix}.ora"
        path, _ = QFileDialog.getSaveFileName(
            self, "OpenRaster로 내보내기", default, "OpenRaster (*.ora)"
        )
        if not path:
            return
        if not path.lower().endswith(".ora"):
            path += ".ora"
        try:
            export_ora(page, merged, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "내보내기 실패", f"ORA로 내보낼 수 없습니다:\n{exc}")
            return
        self.status.showMessage(f"ORA 저장됨: {path}", 4000)

    def export_tiff_doc(self):
        doc = self.canvas.doc
        if doc is None:
            QMessageBox.information(self, "내보내기", "먼저 이미지를 여세요.")
            return
        if not self._confirm_before_save():
            return

        # 멀티페이지면 전체를 한 TIFF로 묶을지 현재 페이지만 할지 선택
        all_pages = True
        if doc.page_count > 1:
            res = QMessageBox.question(
                self, "TIFF 내보내기",
                f"전체 {doc.page_count}페이지를 하나의 멀티페이지 TIFF로 저장할까요?\n"
                "('아니오'를 누르면 현재 페이지만 저장합니다)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if res == QMessageBox.StandardButton.Cancel:
                return
            all_pages = res == QMessageBox.StandardButton.Yes

        pages = doc.pages if all_pages else [doc.current_page]
        flats = [flatten_page(p) for p in pages]

        default = "export.tif"
        if doc.path:
            stem = os.path.splitext(os.path.basename(doc.path))[0]
            suffix = "" if all_pages else f"_p{doc.current_index + 1}"
            default = f"{stem}{suffix}_marked.tif"
        path, _ = QFileDialog.getSaveFileName(
            self, "TIFF로 내보내기", default, "TIFF 이미지 (*.tif *.tiff)"
        )
        if not path:
            return
        if not path.lower().endswith((".tif", ".tiff")):
            path += ".tif"
        try:
            export_tiff(flats, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "내보내기 실패", f"TIFF로 내보낼 수 없습니다:\n{exc}")
            return
        self.status.showMessage(f"TIFF 저장됨({len(flats)}페이지): {path}", 4000)

    def export_pdf_doc(self):
        doc = self.canvas.doc
        if doc is None:
            QMessageBox.information(self, "내보내기", "먼저 이미지를 여세요.")
            return
        if not self._confirm_before_save():
            return
        default = "export.pdf"
        if doc.path:
            default = os.path.splitext(os.path.basename(doc.path))[0] + "_layers.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "PDF로 내보내기(레이어 토글)", default, "PDF 문서 (*.pdf)"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            export_pdf(doc.pages, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "내보내기 실패", f"PDF로 내보낼 수 없습니다:\n{exc}")
            return
        self.status.showMessage(f"PDF 저장됨({doc.page_count}페이지): {path}", 4000)

    # ---------- 마킹검사(OCR) ----------
    def _set_ocr_engine(self, key: str):
        ok, reason = engine_available(key)
        if not ok:
            QMessageBox.warning(self, "OCR 엔진 사용 불가", reason)
            # 현재 엔진 선택 상태로 되돌림
            self._engine_actions[self._ocr_engine].setChecked(True)
            return
        if key != self._ocr_engine:
            self._ocr_engine = key
            self._ocr_cache.clear()  # 엔진 바뀌면 재검출
            self.canvas.clear_overlay()
        self.status.showMessage(f"OCR 엔진: {key}", 3000)

    def _set_ocr_device(self, key: str):
        if key == DEVICE_GPU and not cuda_available():
            QMessageBox.information(
                self, tr("GPU 사용 불가", "GPU unavailable"),
                tr("CUDA를 사용할 수 없어 CPU로 동작합니다.",
                   "CUDA is not available; using CPU instead."),
            )
        set_device(key)
        self._ocr_cache.clear()  # 장치 바뀌면 재검출(속도 차 반영)
        self.canvas.clear_overlay()
        label = {DEVICE_AUTO: tr("자동", "Auto"), DEVICE_GPU: "GPU (CUDA)",
                 DEVICE_CPU: "CPU"}.get(key, key)
        self.status.showMessage(tr("OCR 연산 장치: ", "OCR device: ") + label, 3000)

    def _set_ui_lang(self, lang: str):
        set_ui_lang(lang)
        self._retranslate()
        if lang in self._uilang_actions:
            self._uilang_actions[lang].setChecked(True)

    # ---------- 보안 모드 ----------
    def _security_log_path(self) -> str:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, "security_block.log")

    def _sec_mode(self) -> str:
        return security.MODE_WARN if self._security_warn else security.MODE_QUIT

    def _on_toggle_security(self, checked: bool):
        self._security_on = checked
        if checked:
            security.enable(self._security_log_path(), self._sec_mode())
            self._sec_seen = security.violation_count()
            mode_txt = (tr("외부 연결을 막되 종료하지 않고 경고만 표시합니다.",
                           "External connections are blocked (no quit); a warning is shown.")
                        if self._security_warn else
                        tr("외부로 연결을 시도하면 데이터 전송 전에 즉시 강제종료됩니다.",
                           "Any external connection attempt force-quits before any data is sent."))
            QMessageBox.information(
                self, tr("보안 모드 켜짐", "Security mode ON"),
                tr("외부 네트워크 통신을 차단합니다.\n", "Outbound network is blocked.\n") + mode_txt)
        else:
            security.disable()
        self._update_security_badge()

    def _on_toggle_security_warn(self, checked: bool):
        self._security_warn = checked
        if self._security_on:
            security.set_mode(self._sec_mode())
        self._update_security_badge()

    def _apply_security_state(self):
        """저장된 보안 모드 상태를 다이얼로그 없이 반영(시작 시)."""
        if self._security_on:
            security.enable(self._security_log_path(), self._sec_mode())
            self._sec_seen = security.violation_count()
        for act, val in ((self.act_security, self._security_on),
                         (self.act_security_warn, self._security_warn)):
            act.blockSignals(True)
            act.setChecked(val)
            act.blockSignals(False)
        self._update_security_badge()

    def _update_security_badge(self):
        if not hasattr(self, "security_label"):
            return
        if not self._security_on:
            self.security_label.setText("")
            self.security_label.setStyleSheet("")
            return
        suffix = tr(" · 경고", " · warn") if self._security_warn else ""
        cnt = security.violation_count()
        blocked = (tr(" · 차단 ", " · blocked ") + str(cnt)) if cnt else ""
        self.security_label.setText(tr("🔒 보안", "🔒 Secure") + suffix + blocked)
        self.security_label.setStyleSheet(
            "color: white; background:#c0392b; font-weight:bold; padding:0 8px;")

    def _poll_security(self):
        """경고 모드에서 새 위반 발생 시 상태바·뱃지에 표시."""
        if not self._security_on:
            return
        cnt = security.violation_count()
        if cnt > self._sec_seen:
            self._sec_seen = cnt
            v = security.last_violation()
            addr = v[2] if v else "?"
            self.status.showMessage(
                tr(f"⚠ 외부 통신 차단됨: {addr}", f"⚠ External connection blocked: {addr}"), 8000)
            self._update_security_badge()

    def _retranslate(self):
        """언어 변경 시 메뉴·툴바를 다시 그려 즉시 반영한다."""
        self.menuBar().clear()
        for tb in getattr(self, "_toolbars", []):
            self.removeToolBar(tb)
            tb.deleteLater()
        self._build_actions()
        self._build_toolbar()
        # 상태/표시 갱신
        self._refresh_layer_combo()
        self._update_undo_actions()
        self._update_status()
        self._update_security_badge()

    def _cache_key(self, page):
        return (id(page), self._ocr_engine, self._ocr_langs)

    def _get_boxes(self, page):
        """캐시된 OCR 박스를 반환(숫자·치수 필터 적용). 미검출이면 None."""
        raw = self._ocr_cache.get(self._cache_key(page))
        if raw is None:
            return None
        if self._digits_only:
            return [b for b in raw if any(ch.isdigit() for ch in b.text)]
        return raw

    def _on_toggle_digits_only(self, checked: bool):
        self._digits_only = checked
        # 재검출 불필요(읽기 시 필터) — 표시만 갱신
        page = self.canvas.page
        if page is not None and self._get_boxes(page) is not None:
            unmarked = find_unmarked(page, self._get_boxes(page))
            self.canvas.set_overlay_rects([it.rect for it in unmarked])
            self._nav_unmarked = unmarked
            self._nav_idx = -1
        self._update_progress()

    def _ensure_boxes(self, pages, title: str) -> bool:
        """필요한 페이지의 OCR을 백그라운드로 수행(진행률·취소). 완료 True/취소 False.

        실패 시 예외를 던진다. 이미 캐시된 페이지뿐이면 즉시 True.
        """
        todo = [p for p in pages if self._cache_key(p) not in self._ocr_cache]
        if not todo:
            return True
        dlg = QProgressDialog(title, "취소", 0, len(todo), self)
        dlg.setWindowTitle("마킹검사 (OCR)")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        worker = _OcrWorker(todo, self._ocr_engine, self._ocr_langs, self._ocr_cache)
        err: dict = {}
        worker.progressed.connect(dlg.setValue)
        worker.failed.connect(lambda m: err.update(msg=m))
        dlg.canceled.connect(worker.cancel)
        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        worker.start()
        loop.exec()
        worker.wait()
        dlg.close()
        if err:
            raise RuntimeError(err["msg"])
        return all(self._cache_key(p) in self._ocr_cache for p in pages)

    def _scan_pages(self, pages):
        """여러 페이지 OCR 검사 → (페이지인덱스, 미마킹) 목록. 취소 시 None."""
        t0 = time.perf_counter()
        if not self._ensure_boxes(pages, "OCR 마킹검사 중… (첫 실행은 모델 로딩으로 느릴 수 있음)"):
            return None
        result = []
        for pi, page in enumerate(pages):
            boxes = self._get_boxes(page)
            for it in find_unmarked(page, boxes):
                result.append((pi, it))
        elapsed = time.perf_counter() - t0
        self.status.showMessage(
            f"마킹검사 완료 · 엔진 {self._ocr_engine} · {elapsed:.2f}초 "
            f"· {len(pages)}페이지", 8000
        )
        return result

    def check_marks(self):
        """수동 마킹검사: 현재 페이지의 미마킹 항목을 표시한다."""
        page = self.canvas.page
        if page is None:
            QMessageBox.information(self, tr("마킹검사", "Mark Check"),
                                    tr("먼저 이미지를 여세요.", "Open an image first."))
            return
        cached = self._cache_key(page) in self._ocr_cache
        t0 = time.perf_counter()
        try:
            if not self._ensure_boxes([page], tr("OCR 마킹검사 중…", "Running OCR…")):
                return  # 취소
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, tr("마킹검사 실패", "Mark check failed"),
                                 tr("OCR 실행 중 오류:\n", "OCR error:\n") + str(exc))
            return
        boxes = self._get_boxes(page)
        elapsed = time.perf_counter() - t0
        note = tr("캐시", "cached") if cached else f"{elapsed:.2f}s"
        self.status.showMessage(tr(
            f"마킹검사 · 엔진 {self._ocr_engine} · {note} · 텍스트 {len(boxes)}개",
            f"Mark check · {self._ocr_engine} · {note} · {len(boxes)} texts"), 8000)
        unmarked = find_unmarked(page, boxes)
        self.canvas.set_overlay_rects([it.rect for it in unmarked])
        self._nav_unmarked = unmarked  # 네비게이션용 저장
        self._nav_idx = -1
        self._update_progress()
        if not unmarked:
            QMessageBox.information(
                self, tr("마킹검사", "Mark Check"),
                tr(f"텍스트 {len(boxes)}개 모두 마킹되어 있습니다. ✅",
                   f"All {len(boxes)} text items are marked. ✅"))
            return
        sample = "\n".join(f"  • {it.text}" for it in unmarked[:15])
        more = "" if len(unmarked) <= 15 else tr(
            f"\n  … 외 {len(unmarked) - 15}개", f"\n  … and {len(unmarked) - 15} more")
        QMessageBox.warning(
            self, tr("미마킹 항목 발견", "Unmarked items found"),
            tr(f"마커가 없는 텍스트 {len(unmarked)}개를 찾았습니다(전체 {len(boxes)}개).\n"
               f"빨간 점선으로 표시했습니다. 'N' 키로 순회할 수 있습니다.\n\n{sample}{more}",
               f"Found {len(unmarked)} unmarked text items (of {len(boxes)}).\n"
               f"Shown with red dashes. Press 'N' to cycle through them.\n\n{sample}{more}"))

    def goto_next_unmarked(self):
        """미마킹 항목을 차례로 화면 중앙에 보여준다."""
        if self.canvas.page is None:
            return
        if not self._nav_unmarked:
            self.check_marks()  # 아직 검사 안 했으면 먼저 검사
            if not self._nav_unmarked:
                return
        self._nav_idx = (self._nav_idx + 1) % len(self._nav_unmarked)
        it = self._nav_unmarked[self._nav_idx]
        self.canvas.center_on_image_rect(it.rect)
        self.canvas.set_overlay_rects([u.rect for u in self._nav_unmarked])
        self.status.showMessage(
            f"미마킹 {self._nav_idx + 1}/{len(self._nav_unmarked)}: {it.text}", 6000
        )

    # ---------- 자동 마킹 ----------
    def _box_already_marked(self, layer, rect, ratio: float = 0.25) -> bool:
        """레이어의 해당 영역이 이미 (마커로) 칠해져 있는지 대략 판정."""
        img = layer.image
        x0, y0 = max(0, rect.left()), max(0, rect.top())
        x1 = min(img.width(), rect.left() + rect.width())
        y1 = min(img.height(), rect.top() + rect.height())
        if x1 <= x0 or y1 <= y0:
            return False
        total = hit = 0
        step = max(1, (x1 - x0) // 12)
        ystep = max(1, (y1 - y0) // 6)
        for yy in range(y0, y1, ystep):
            for xx in range(x0, x1, step):
                total += 1
                if img.pixelColor(xx, yy).alpha() > 0:
                    hit += 1
        return total > 0 and hit / total >= ratio

    def _on_automark(self, img_pt: QPointF):
        """자동마킹 도구: 클릭 위치의 검출 박스를 형광펜으로 채운다."""
        page = self.canvas.page
        if page is None or not page.active_layer.visible:
            return
        key = self._cache_key(page)
        if key not in self._ocr_cache:
            try:
                if not self._ensure_boxes([page], "OCR 검출 중… (자동마킹 준비)"):
                    return  # 취소
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "자동마킹 실패", f"OCR 실행 중 오류:\n{exc}")
                return
        boxes = self._get_boxes(page) or []
        pt = QPoint(int(img_pt.x()), int(img_pt.y()))
        # 점을 포함하는 가장 작은 박스 선택
        cands = [it for it in boxes if it.rect.contains(pt)]
        if not cands:
            return
        it = min(cands, key=lambda b: b.rect.width() * b.rect.height())
        if self._box_already_marked(page.active_layer, it.rect):
            return  # 이미 칠해진 박스는 건너뜀(중복 방지)
        self.canvas.mark_box_highlight(it.rect)
        self._update_undo_actions()

    def _confirm_before_save(self) -> bool:
        """저장 전 자동 마킹검사. 미마킹이 있으면 계속/취소를 묻는다. 진행하면 True."""
        if not self._autocheck_on or self.canvas.doc is None:
            return True
        try:
            unmarked = self._scan_pages(self.canvas.doc.pages)
        except Exception as exc:  # noqa: BLE001
            # OCR 실패가 저장을 막지 않도록 경고만
            QMessageBox.warning(self, "마킹검사 건너뜀", f"OCR 검사 실패:\n{exc}")
            return True
        if unmarked is None:
            return True  # 사용자가 검사를 취소 → 저장은 진행
        if not unmarked:
            return True
        # 현재 페이지의 미마킹을 화면에 표시
        cur = self.canvas.doc.current_index
        self.canvas.set_overlay_rects(
            [it.rect for pi, it in unmarked if pi == cur]
        )
        by_page = {}
        for pi, it in unmarked:
            by_page.setdefault(pi, []).append(it.text)
        lines = []
        for pi in sorted(by_page):
            texts = ", ".join(by_page[pi][:8])
            extra = "" if len(by_page[pi]) <= 8 else " …"
            lines.append(f"  p{pi + 1}: {len(by_page[pi])}개 — {texts}{extra}")
        detail = "\n".join(lines)
        res = QMessageBox.warning(
            self, "미마킹 항목 있음",
            f"마커가 없는 텍스트 {len(unmarked)}개가 있습니다.\n\n{detail}\n\n"
            "그래도 저장할까요?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return res == QMessageBox.StandardButton.Yes

    def export_png(self):
        if not self._confirm_before_save():
            return
        flat = self.canvas.render_flat()
        if flat is None:
            QMessageBox.information(self, "내보내기", "먼저 이미지를 여세요.")
            return
        doc = self.canvas.doc
        default = "export.png"
        if doc and doc.path:
            stem = os.path.splitext(os.path.basename(doc.path))[0]
            suffix = f"_p{doc.current_index + 1}" if doc.page_count > 1 else ""
            default = f"{stem}{suffix}_marked.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "PNG로 내보내기", default, "PNG 이미지 (*.png)"
        )
        if not path:
            return
        if flat.save(path, "PNG"):
            self.status.showMessage(f"저장됨: {path}", 4000)
        else:
            QMessageBox.critical(self, "저장 실패", "PNG 저장에 실패했습니다.")
