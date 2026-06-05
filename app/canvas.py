"""그리기 캔버스 위젯: 줌/팬, 펜·형광펜·지우개, 레이어 합성, 실행취소."""
from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from .layer import (
    Document,
    Page,
    Stroke,
    build_automark_stroke,
    composition_mode,
    new_transparent_image,
    paint_stroke,
    stroke_hit,
)
from .layer import STROKE_TEXT

TOOL_PEN = "pen"
TOOL_HIGHLIGHTER = "highlighter"
TOOL_ERASER = "eraser"
TOOL_AUTOMARK = "automark"  # 검출된 치수 박스를 클릭/드래그로 자동 형광 마킹
TOOL_TEXT = "text"          # 클릭 위치에 텍스트(글자) 입력
TOOL_HAND = "hand"          # 손바닥 도구: 좌클릭 드래그로 화면 이동(팬)

ERASE_MODE_STROKE = "stroke"  # 획 전체 지우기
ERASE_MODE_PIXEL = "pixel"    # 부분(픽셀) 지우기

WIDTH_PX = "px"    # 굵기 = 이미지 픽셀(절대값)
WIDTH_PCT = "pct"  # 굵기 = 이미지 폭의 %

MIN_SCALE = 0.05
MAX_SCALE = 16.0
UNDO_LIMIT = 40


def _default_widths() -> dict:
    # 도구별·단위별 굵기를 독립 저장(전환해도 서로 영향 없음)
    return {
        TOOL_PEN: {WIDTH_PX: 3.0, WIDTH_PCT: 0.30},
        TOOL_HIGHLIGHTER: {WIDTH_PX: 18.0, WIDTH_PCT: 1.50},
        TOOL_ERASER: {WIDTH_PX: 24.0, WIDTH_PCT: 2.00},
        TOOL_AUTOMARK: {WIDTH_PX: 1.0, WIDTH_PCT: 0.10},  # 미사용(스핀 호환용)
        TOOL_HAND: {WIDTH_PX: 1.0, WIDTH_PCT: 0.10},      # 미사용(스핀 호환용)
        TOOL_TEXT: {WIDTH_PX: 24.0, WIDTH_PCT: 2.00},     # 글자 크기(픽셀/이미지%)
    }


@dataclass
class ToolSettings:
    tool: str = TOOL_PEN
    pen_color: QColor = None
    highlighter_color: QColor = None
    highlighter_opacity: float = 0.40
    eraser_mode: str = ERASE_MODE_STROKE
    width_mode: str = WIDTH_PCT  # 굵기 단위 기본값: 이미지 폭 %
    widths: dict = field(default_factory=_default_widths)
    automark_grow: int = 2  # 자동마킹(글자 모양만): 글자 마스크 팽창 px
    automark_box: bool = True  # 자동마킹 채우기: True=텍스트 박스 전체, False=글자 모양만

    def __post_init__(self):
        if self.pen_color is None:
            self.pen_color = QColor(220, 30, 30)
        if self.highlighter_color is None:
            self.highlighter_color = QColor(255, 235, 0)

    def get_width(self, tool: str) -> float:
        """현재 단위에서의 해당 도구 굵기 값."""
        return self.widths[tool][self.width_mode]

    def set_width(self, tool: str, value: float):
        self.widths[tool][self.width_mode] = value

    def px_width(self, tool: str, page_width: int) -> float:
        """해당 도구의 실제 이미지 픽셀 두께(단위·페이지 폭 반영)."""
        v = self.widths[tool][self.width_mode]
        if self.width_mode == WIDTH_PCT and page_width > 0:
            return max(0.5, v / 100.0 * page_width)
        return v


class _UndoEntry:
    """한 동작 직전의 특정 레이어 상태(이미지 + 획 목록) 스냅샷."""

    def __init__(self, page: Page, layer_index: int, image: QImage,
                 strokes: list[Stroke]):
        self.page = page
        self.layer_index = layer_index
        self.image = image
        self.strokes = strokes


class Canvas(QWidget):
    layersChanged = Signal()       # 레이어 구성/속성 변경 → 패널 갱신
    viewChanged = Signal()         # 줌/페이지 변경 → 상태바 갱신
    undoStateChanged = Signal()    # 실행취소 가능 여부 변경
    autoMarkRequested = Signal(QPointF)  # 자동마킹 도구 클릭/드래그 위치(이미지 좌표)
    autoMarkStarted = Signal()           # 자동마킹 한 번의 드래그 시작(중복방지 세션 초기화)
    textRequested = Signal(QPointF)      # 텍스트 도구 클릭 위치(이미지 좌표)
    contentChanged = Signal()      # 활성 레이어 그림 변경(마킹 진척도 갱신용)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self.doc: Document | None = None
        self.tools = ToolSettings()

        self.scale = 1.0
        self.offset = QPointF(0, 0)  # 뷰 좌표에서 이미지(0,0)의 위치

        # 합성 캐시(구조 변경 시에만 재생성)
        self._below: QImage | None = None
        self._above: QImage | None = None

        # 마킹검사: 미마킹 텍스트 박스 오버레이(현재 페이지, 이미지 좌표)
        self._overlay_rects: list = []

        # 텍스트 붙여넣기 미리보기(커서를 따라다니는 반투명 글자)
        self._text_preview = ""
        self._preview_img_pt: QPointF | None = None
        self._preview_angle = 0.0  # 미리보기 회전 각도(도, 시계방향 +)

        # 캔버스 크기 조절(그림판식): 모서리/가장자리 핸들 드래그
        self._resizing: str | None = None       # 'e' | 's' | 'se'
        self._resize_size: tuple | None = None   # 드래그 중 미리보기 크기 (w, h)

        # 그리는 중 상태
        self._drawing = False
        self._erasing = False
        self._automarking = False
        self._last_img_pt: QPointF | None = None
        self._stroke_buffer: QImage | None = None  # 형광펜 한 획 누적용
        self._current_points: list[QPointF] = []   # 현재 그리는 획의 점들

        # 패닝 상태
        self._panning = False
        self._pan_last: QPoint | None = None

        # 실행취소/다시실행
        self._undo: list[_UndoEntry] = []
        self._redo: list[_UndoEntry] = []

    # ---------- 문서 관리 ----------
    def set_document(self, doc: Document):
        self.doc = doc
        self._undo.clear()
        self._redo.clear()
        self._overlay_rects = []
        self._rebuild_caches()
        self.fit_to_window()
        self.layersChanged.emit()
        self.viewChanged.emit()
        self.undoStateChanged.emit()

    @property
    def page(self) -> Page | None:
        return self.doc.current_page if self.doc else None

    def _px_width(self, tool: str) -> float:
        """도구의 현재 단위/페이지 폭 기준 실제 픽셀 두께."""
        pw = self.page.size.width() if self.page else 0
        return self.tools.px_width(tool, pw)

    # ---------- 합성 캐시 ----------
    def _rebuild_caches(self):
        """현재 페이지에서 활성 레이어 아래/위를 각각 평탄화해 캐시한다."""
        page = self.page
        if page is None:
            self._below = self._above = None
            return
        size = page.size
        # 현재 모드(그룹) 레이어만 합성 → 다른 모드 마킹은 완전히 숨김
        cur = page.current_layers()
        active = page.active_layer
        ai = cur.index(active) if active in cur else len(cur)

        below = new_transparent_image(size)
        bp = QPainter(below)
        bp.fillRect(below.rect(), Qt.GlobalColor.white)
        bp.drawImage(0, 0, page.base)
        for layer in cur[:ai]:
            if not layer.visible:
                continue
            bp.setOpacity(layer.opacity)
            bp.setCompositionMode(composition_mode(layer.blend))
            bp.drawImage(0, 0, layer.image)
        bp.end()
        self._below = below

        above = new_transparent_image(size)
        ap = QPainter(above)
        for layer in cur[ai + 1:]:
            if not layer.visible:
                continue
            ap.setOpacity(layer.opacity)
            ap.setCompositionMode(composition_mode(layer.blend))
            ap.drawImage(0, 0, layer.image)
        ap.end()
        self._above = above

    def notify_layers_changed(self):
        """외부(레이어 패널)에서 구조/속성을 바꾼 뒤 호출."""
        self._rebuild_caches()
        self.update()
        self.layersChanged.emit()

    def _active_render(self) -> QImage:
        """활성 레이어 + 진행 중인 형광펜 획 미리보기를 합친 이미지."""
        page = self.page
        active = page.active_layer
        if self._stroke_buffer is None:
            return active.image
        merged = QImage(active.image)  # 얕은 공유 → painter가 detach
        p = QPainter(merged)
        p.setOpacity(self.tools.highlighter_opacity)
        p.drawImage(0, 0, self._stroke_buffer)
        p.end()
        return merged

    # ---------- 좌표 변환 ----------
    def view_to_image(self, pt: QPointF) -> QPointF:
        return QPointF((pt.x() - self.offset.x()) / self.scale,
                       (pt.y() - self.offset.y()) / self.scale)

    def image_to_view(self, x: float, y: float) -> QPointF:
        return QPointF(self.offset.x() + x * self.scale,
                       self.offset.y() + y * self.scale)

    # ---------- 캔버스 크기 조절 핸들 ----------
    _HANDLE = 5  # 핸들 반쪽 크기(px)
    _HANDLE_TOL = 9  # 클릭 허용 반경(px)
    MIN_CANVAS = 16  # 최소 캔버스 크기

    def _resize_handles(self) -> dict:
        """오른쪽(e)·아래(s)·우하단 모서리(se) 핸들의 뷰 좌표."""
        if self.page is None:
            return {}
        w, h = self.page.size.width(), self.page.size.height()
        return {
            "se": self.image_to_view(w, h),
            "e": self.image_to_view(w, h / 2),
            "s": self.image_to_view(w / 2, h),
        }

    def _handle_at(self, view_pt: QPointF) -> str | None:
        tol = self._HANDLE_TOL
        for key, h in self._resize_handles().items():  # se 먼저(모서리 우선)
            if abs(view_pt.x() - h.x()) <= tol and abs(view_pt.y() - h.y()) <= tol:
                return key
        return None

    def _update_resize(self, view_pt: QPointF):
        img = self.view_to_image(view_pt)
        w, h = self._resize_size
        if self._resizing in ("e", "se"):
            w = max(self.MIN_CANVAS, int(round(img.x())))
        if self._resizing in ("s", "se"):
            h = max(self.MIN_CANVAS, int(round(img.y())))
        self._resize_size = (w, h)
        self.update()

    def _finish_resize(self):
        size = self._resize_size
        self._resizing = None
        self._resize_size = None
        page = self.page
        if page is not None and size and (
                size != (page.size.width(), page.size.height())):
            page.resize_canvas(size[0], size[1])
            self._undo.clear()        # 이전 크기 스냅샷은 무효
            self._redo.clear()
            self.undoStateChanged.emit()
            self._rebuild_caches()
            self.viewChanged.emit()   # 스크롤바/상태 갱신
            self.contentChanged.emit()
        self.update()
        self.apply_tool_cursor()

    # ---------- 줌/팬 ----------
    def fit_to_window(self):
        page = self.page
        if page is None:
            return
        size = page.size
        if size.width() == 0 or size.height() == 0:
            return
        margin = 20
        sx = (self.width() - margin) / size.width()
        sy = (self.height() - margin) / size.height()
        self.scale = max(MIN_SCALE, min(MAX_SCALE, min(sx, sy)))
        self._center_image()
        self.update()
        self.viewChanged.emit()

    def set_scale(self, scale: float, anchor: QPointF | None = None):
        scale = max(MIN_SCALE, min(MAX_SCALE, scale))
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)
        img_pt = self.view_to_image(anchor)
        self.scale = scale
        # anchor 아래 이미지 점이 그대로 유지되도록 offset 보정
        self.offset = QPointF(anchor.x() - img_pt.x() * scale,
                              anchor.y() - img_pt.y() * scale)
        self.update()
        self.viewChanged.emit()

    def scroll_metrics(self):
        """스크롤바 동기화용 (가로max, 가로페이지, 가로값, 세로max, 세로페이지, 세로값).

        값은 뷰 픽셀 기준. 문서가 없으면 None.
        """
        if self.page is None:
            return None
        iw = self.page.size.width() * self.scale
        ih = self.page.size.height() * self.scale
        vw, vh = self.width(), self.height()
        hmax, vmax = max(0, int(iw - vw)), max(0, int(ih - vh))
        hval = int(min(hmax, max(0, -self.offset.x())))
        vval = int(min(vmax, max(0, -self.offset.y())))
        return (hmax, int(vw), hval, vmax, int(vh), vval)

    def set_scroll(self, hval: int | None = None, vval: int | None = None):
        """스크롤 위치를 뷰 픽셀 값으로 설정(범위 안으로 클램프). 이미지가 뷰보다
        작은 축은 중앙 정렬을 유지한다."""
        m = self.scroll_metrics()
        if m is None:
            return
        hmax, _, hcur, vmax, _, vcur = m
        h = hcur if hval is None else max(0, min(hmax, hval))
        v = vcur if vval is None else max(0, min(vmax, vval))
        if hmax > 0:
            self.offset.setX(-float(h))
        if vmax > 0:
            self.offset.setY(-float(v))
        self.update()
        self.viewChanged.emit()

    def pan_by(self, dx: float, dy: float):
        """현재 위치에서 화면을 (dx, dy) 뷰 픽셀만큼 이동(방향키용, 범위 클램프)."""
        m = self.scroll_metrics()
        if m is None:
            return
        # dx>0 = 왼쪽 내용 보기(이미지 오른쪽으로). 스크롤값은 그 반대.
        self.set_scroll(hval=m[2] - int(dx), vval=m[5] - int(dy))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.viewChanged.emit()  # 스크롤바 범위 재계산

    def apply_tool_cursor(self):
        """현재 도구에 맞는 커서를 적용(손 도구는 손바닥 모양)."""
        if self.tools.tool == TOOL_HAND:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _center_image(self):
        page = self.page
        if page is None:
            return
        size = page.size
        self.offset = QPointF(
            (self.width() - size.width() * self.scale) / 2,
            (self.height() - size.height() * self.scale) / 2,
        )

    # ---------- 페인트 ----------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(60, 60, 64))
        if self.doc is None or self._below is None:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "파일 → 열기 로 이미지를 불러오세요")
            painter.end()
            return

        page = self.page
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                              self.scale < 1.0)
        painter.translate(self.offset)
        painter.scale(self.scale, self.scale)

        # 아래 캐시
        painter.drawImage(0, 0, self._below)
        # 활성 레이어
        active = page.active_layer
        if active.visible:
            painter.setOpacity(active.opacity)
            painter.setCompositionMode(composition_mode(active.blend))
            painter.drawImage(0, 0, self._active_render())
            painter.setOpacity(1.0)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        # 위 캐시
        painter.drawImage(0, 0, self._above)

        # 미마킹 항목 오버레이(빨간 점선 박스)
        if self._overlay_rects:
            pen = QPen(QColor(230, 30, 30), 2)
            pen.setCosmetic(True)  # 줌과 무관하게 일정 두께
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for r in self._overlay_rects:
                painter.drawRect(r)

        # 텍스트 붙여넣기 미리보기(이미지 좌표계 그대로 → 줌에 맞춰 크기 반영)
        self._paint_text_preview(painter)

        # 캔버스 크기 조절 핸들/미리보기(뷰 좌표 → 변환 초기화 후 그림)
        painter.resetTransform()
        if self._resizing and self._resize_size:
            w, h = self._resize_size
            tl = self.image_to_view(0, 0)
            br = self.image_to_view(w, h)
            pen = QPen(QColor(30, 120, 230), 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(tl, br))
        for hp in self._resize_handles().values():
            painter.setPen(QPen(QColor(40, 40, 40), 1))
            painter.setBrush(QColor(255, 255, 255))
            s = self._HANDLE
            painter.drawRect(QRectF(hp.x() - s, hp.y() - s, 2 * s, 2 * s))
        painter.end()

    def set_text_preview(self, text: str, angle: float = 0.0):
        """붙여넣기 대기 텍스트를 커서 옆에 미리 보여준다('' 면 끔)."""
        self._text_preview = text or ""
        self._preview_angle = angle
        self.update()

    def set_text_preview_angle(self, angle: float):
        if self._text_preview:
            self._preview_angle = angle
            self.update()

    def clear_text_preview(self):
        if self._text_preview:
            self._text_preview = ""
            self.update()

    def _paint_text_preview(self, painter: QPainter):
        """현재 커서 위치(이미지 좌표)에 반투명 글자 미리보기를 그린다(회전 반영)."""
        if not self._text_preview or self._preview_img_pt is None:
            return
        font = QFont()
        font.setPixelSize(max(4, int(round(self._px_width(TOOL_TEXT)))))
        painter.setFont(font)
        c = self.tools.pen_color
        painter.setPen(QColor(c.red(), c.green(), c.blue(), 130))  # 반투명
        from PySide6.QtGui import QFontMetrics
        fm = QFontMetrics(font)
        x, y = self._preview_img_pt.x(), self._preview_img_pt.y()
        lh, asc = fm.height(), fm.ascent()
        painter.save()
        painter.translate(x, y)
        if self._preview_angle:
            painter.rotate(self._preview_angle)
        for i, line in enumerate(self._text_preview.split("\n")):
            painter.drawText(0, asc + i * lh, line)
        painter.restore()

    def set_overlay_rects(self, rects: list):
        self._overlay_rects = list(rects)
        self.update()

    def clear_overlay(self):
        if self._overlay_rects:
            self._overlay_rects = []
            self.update()

    # ---------- 입력 ----------
    def wheelEvent(self, event: QWheelEvent):
        if self.doc is None:
            return
        factor = 1.0015 ** event.angleDelta().y()
        self.set_scale(self.scale * factor, QPointF(event.position()))

    def mousePressEvent(self, event: QMouseEvent):
        if self.doc is None:
            return
        left = event.button() == Qt.MouseButton.LeftButton
        # 크기 조절 핸들 위에서 좌클릭(Ctrl 없이) → 캔버스 리사이즈 시작
        if left and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            handle = self._handle_at(event.position())
            if handle is not None:
                self._resizing = handle
                self._resize_size = (self.page.size.width(), self.page.size.height())
                return
        if (event.button() == Qt.MouseButton.MiddleButton
                or (left and event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                or (left and self.tools.tool == TOOL_HAND)):
            self._panning = True
            self._pan_last = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            img_pt = self.view_to_image(QPointF(event.position()))
            if self.tools.tool == TOOL_TEXT:
                self.textRequested.emit(img_pt)  # 입력창은 외부(메인윈도우)에서
            elif self.tools.tool == TOOL_AUTOMARK:
                self._automarking = True
                self.autoMarkStarted.emit()          # 새 드래그 시작 → 세션 초기화
                self.autoMarkRequested.emit(img_pt)  # 박스 채우기는 외부(메인윈도우)에서
            elif (self.tools.tool == TOOL_ERASER
                    and self.tools.eraser_mode == ERASE_MODE_STROKE):
                self._begin_erase(img_pt)   # 획 전체 지우기
            else:
                self._begin_stroke(img_pt)  # 펜/형광펜 또는 부분 지우기

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._resizing:
            self._update_resize(event.position())
            return
        if self._panning and self._pan_last is not None:
            now = event.position().toPoint()
            self.offset += QPointF(now - self._pan_last)
            self._pan_last = now
            self.update()
            return
        # 유휴 상태: 크기 조절 핸들 위면 방향 커서 표시
        if not (self._drawing or self._erasing or self._automarking):
            hk = self._handle_at(event.position())
            if hk is not None:
                self.setCursor({
                    "e": Qt.CursorShape.SizeHorCursor,
                    "s": Qt.CursorShape.SizeVerCursor,
                    "se": Qt.CursorShape.SizeFDiagCursor,
                }[hk])
            elif not self._text_preview:
                self.apply_tool_cursor()
        img_pt = self.view_to_image(QPointF(event.position()))
        if self._text_preview:  # 붙여넣기 미리보기를 커서 위치로 갱신
            self._preview_img_pt = img_pt
            self.update()
        if self._drawing:
            self._continue_stroke(img_pt)
        elif self._erasing:
            self._erase_at(img_pt)
        elif self._automarking:
            self.autoMarkRequested.emit(img_pt)

    def leaveEvent(self, event):
        if self._text_preview and self._preview_img_pt is not None:
            self._preview_img_pt = None  # 캔버스 밖이면 미리보기 숨김
            self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._resizing:
            self._finish_resize()
            return
        if self._panning:
            self._panning = False
            self._pan_last = None
            self.apply_tool_cursor()
            return
        if self._drawing:
            self._end_stroke()
        elif self._erasing:
            self._erasing = False
        elif self._automarking:
            self._automarking = False

    def add_text(self, img_pt: QPointF, text: str, angle: float = 0.0,
                 boxed: bool = False) -> bool:
        """클릭 위치(좌상단 기준)에 텍스트를 활성 레이어에 그린다(angle: 시계방향 도).

        boxed=True면 글자 둘레에 테두리 박스(스탬프/도장).
        """
        page = self.page
        if page is None or not page.active_layer.visible or not text.strip():
            return False
        self._push_undo()
        layer = page.active_layer
        c = self.tools.pen_color
        stroke = Stroke(
            tool=STROKE_TEXT,
            color=(c.red(), c.green(), c.blue(), 255),
            width=float(self._px_width(TOOL_TEXT)),
            opacity=1.0,
            points=[(img_pt.x(), img_pt.y())],
            text=text,
            angle=angle,
            boxed=boxed,
        )
        layer.strokes.append(stroke)
        paint_stroke(layer.image, stroke)
        self.update()
        self.contentChanged.emit()
        return True

    # ---------- 자동 마킹 / 화면 이동 ----------
    def mark_box_highlight(self, rect) -> bool:
        """검출된 OCR 박스를 활성 레이어에 형광펜으로 칠한다.

        채우기 방식(self.tools.automark_box):
          - True : 텍스트 바운더리(박스) 전체를 칠함
          - False: 인식된 글자 모양만 따라 칠함(주변 여백/선 안 건드림)
        """
        page = self.page
        if page is None or not page.active_layer.visible:
            return False
        c = self.tools.highlighter_color
        if getattr(self.tools, "automark_box", True):
            # 박스 전체: 굵은 형광펜 선으로 사각형을 채운다. 박스의 '긴 방향'으로
            # 선을 긋고 '짧은 방향'을 선 두께로 써, 가로/세로 어떤 박스든 정확히
            # 채워진다(형광펜 둥근 캡이 박스 끝에 딱 맞도록 끝점을 두께/2만큼 당김).
            xc, yc = rect.center().x(), rect.center().y()
            bw, bh = rect.width(), rect.height()
            if bw >= bh:  # 가로로 긴(보통) 박스 → 수평선, 두께=높이
                w = float(max(2, bh))
                half = w / 2.0
                a, b = rect.left() + half, rect.right() - half
                pts = [(a, yc), (b, yc)] if b >= a else [(xc, yc)]
            else:         # 세로로 긴 박스 → 수직선, 두께=너비
                w = float(max(2, bw))
                half = w / 2.0
                a, b = rect.top() + half, rect.bottom() - half
                pts = [(xc, a), (xc, b)] if b >= a else [(xc, yc)]
            stroke = Stroke(
                tool=TOOL_HIGHLIGHTER,
                color=(c.red(), c.green(), c.blue(), c.alpha()),
                width=w,
                opacity=self.tools.highlighter_opacity,
                points=pts,
            )
        else:
            grow = int(getattr(self.tools, "automark_grow", 2))
            stroke = build_automark_stroke(
                page.base, rect,
                (c.red(), c.green(), c.blue(), c.alpha()),
                self.tools.highlighter_opacity, grow,
            )
        if stroke is None:
            return False
        self._push_undo()
        layer = page.active_layer
        layer.strokes.append(stroke)
        paint_stroke(layer.image, stroke)
        self.update()
        self.contentChanged.emit()
        return True

    def center_on_image_rect(self, rect):
        """현재 줌은 그대로 두고, 사각형이 화면 중앙에 오도록 이동(스크롤)만 한다."""
        if rect.width() <= 0 or rect.height() <= 0:
            return
        cx, cy = rect.center().x(), rect.center().y()
        self.offset = QPointF(self.width() / 2 - cx * self.scale,
                              self.height() / 2 - cy * self.scale)
        self.update()
        self.viewChanged.emit()

    # ---------- 스트로크 ----------
    def _push_undo(self):
        page = self.page
        layer = page.active_layer
        entry = _UndoEntry(page, page.active_index,
                           QImage(layer.image), list(layer.strokes))
        self._undo.append(entry)
        if len(self._undo) > UNDO_LIMIT:
            self._undo.pop(0)
        self._redo.clear()
        self.undoStateChanged.emit()

    # ---------- 펜/형광펜 ----------
    def _begin_stroke(self, img_pt: QPointF):
        page = self.page
        if page is None or not page.active_layer.visible:
            return
        self._overlay_rects = []  # 편집 시 이전 검사 표시 제거
        self._push_undo()
        self._drawing = True
        self._last_img_pt = img_pt
        self._current_points = [img_pt]
        if self.tools.tool == TOOL_HIGHLIGHTER:
            self._stroke_buffer = new_transparent_image(page.size)
        self._draw_segment(img_pt, img_pt)

    def _continue_stroke(self, img_pt: QPointF):
        if self._last_img_pt is None:
            return
        self._draw_segment(self._last_img_pt, img_pt)
        self._last_img_pt = img_pt
        self._current_points.append(img_pt)

    def _draw_segment(self, p0: QPointF, p1: QPointF):
        """그리는 중 라이브 미리보기(획 데이터는 _end_stroke에서 확정)."""
        page = self.page
        tool = self.tools.tool
        if tool == TOOL_HIGHLIGHTER:
            target = self._stroke_buffer
            color = QColor(self.tools.highlighter_color)
            color.setAlpha(255)  # 버퍼엔 불투명, 합성 시 opacity 적용
            width = self._px_width(TOOL_HIGHLIGHTER)
            mode = QPainter.CompositionMode.CompositionMode_Source
        elif tool == TOOL_ERASER:  # 부분(픽셀) 지우기
            target = page.active_layer.image
            color = QColor(0, 0, 0, 255)
            width = self._px_width(TOOL_ERASER)
            mode = QPainter.CompositionMode.CompositionMode_Clear
        else:  # 펜
            target = page.active_layer.image
            color = QColor(self.tools.pen_color)
            width = self._px_width(TOOL_PEN)
            mode = QPainter.CompositionMode.CompositionMode_SourceOver

        painter = QPainter(target)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setCompositionMode(mode)
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        if p0 == p1:
            painter.drawPoint(p0)
        else:
            painter.drawLine(p0, p1)
        painter.end()
        self.update()

    def _end_stroke(self):
        page = self.page
        tool = self.tools.tool
        if tool == TOOL_HIGHLIGHTER and self._stroke_buffer is not None:
            # 한 획 버퍼를 활성 레이어에 opacity 적용해 합성(자기겹침 누적 방지)
            painter = QPainter(page.active_layer.image)
            painter.setOpacity(self.tools.highlighter_opacity)
            painter.drawImage(0, 0, self._stroke_buffer)
            painter.end()

        # 획을 객체로 기록(획 지우개/재렌더/저장에 일관 반영)
        if self._current_points:
            if tool == TOOL_HIGHLIGHTER:
                c = self.tools.highlighter_color
                width = self._px_width(TOOL_HIGHLIGHTER)
                opacity = self.tools.highlighter_opacity
                rgba = (c.red(), c.green(), c.blue(), c.alpha())
            elif tool == TOOL_ERASER:  # 부분 지우개 획
                width = self._px_width(TOOL_ERASER)
                opacity = 1.0
                rgba = (0, 0, 0, 255)
            else:
                c = self.tools.pen_color
                width = self._px_width(TOOL_PEN)
                opacity = 1.0
                rgba = (c.red(), c.green(), c.blue(), c.alpha())
            stroke = Stroke(
                tool=tool,
                color=rgba,
                width=width,
                opacity=opacity,
                points=[(p.x(), p.y()) for p in self._current_points],
            )
            page.active_layer.strokes.append(stroke)

        self._stroke_buffer = None
        self._drawing = False
        self._last_img_pt = None
        self._current_points = []
        self.update()
        self.contentChanged.emit()

    # ---------- 지우개(획 단위) ----------
    def _begin_erase(self, img_pt: QPointF):
        page = self.page
        if page is None or not page.active_layer.visible:
            return
        # 지울 펜/형광펜 획이 하나도 없으면 무시
        if not any(s.tool != TOOL_ERASER for s in page.active_layer.strokes):
            return
        self._push_undo()
        self._erasing = True
        self._erase_at(img_pt)

    def _erase_at(self, img_pt: QPointF):
        """지우개 반경에 닿는 펜/형광펜 획을 통째로 제거한다(지우개 획은 보존)."""
        layer = self.page.active_layer
        tol = self._px_width(TOOL_ERASER) / 2
        px, py = img_pt.x(), img_pt.y()
        kept = [
            s for s in layer.strokes
            if s.tool == TOOL_ERASER or not stroke_hit(s, px, py, tol)
        ]
        if len(kept) != len(layer.strokes):
            layer.strokes = kept
            layer.rerender()
            self.update()
            self.contentChanged.emit()

    # ---------- 실행취소/다시실행 ----------
    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self):
        if not self._undo:
            return
        entry = self._undo.pop()
        layer = entry.page.layers[entry.layer_index]
        self._redo.append(_UndoEntry(entry.page, entry.layer_index,
                                     QImage(layer.image), list(layer.strokes)))
        layer.image = entry.image
        layer.strokes = entry.strokes
        self._rebuild_caches()
        self.update()
        self.undoStateChanged.emit()
        self.contentChanged.emit()

    def redo(self):
        if not self._redo:
            return
        entry = self._redo.pop()
        layer = entry.page.layers[entry.layer_index]
        self._undo.append(_UndoEntry(entry.page, entry.layer_index,
                                     QImage(layer.image), list(layer.strokes)))
        layer.image = entry.image
        layer.strokes = entry.strokes
        self._rebuild_caches()
        self.update()
        self.undoStateChanged.emit()
        self.contentChanged.emit()

    # ---------- 내보내기 ----------
    def render_flat(self) -> QImage | None:
        """현재 페이지를 보이는 레이어까지 합쳐 한 장의 RGB 이미지로 반환."""
        page = self.page
        if page is None:
            return None
        flat = QImage(page.size, QImage.Format.Format_ARGB32_Premultiplied)
        flat.fill(Qt.GlobalColor.white)
        p = QPainter(flat)
        p.drawImage(0, 0, page.base)
        for layer in page.layers:
            if not layer.visible:
                continue
            p.setOpacity(layer.opacity)
            p.setCompositionMode(composition_mode(layer.blend))
            p.drawImage(0, 0, layer.image)
        p.end()
        return flat.convertToFormat(QImage.Format.Format_RGB888)
