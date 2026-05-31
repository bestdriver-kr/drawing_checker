"""그리기 캔버스 위젯: 줌/팬, 펜·형광펜·지우개, 레이어 합성, 실행취소."""
from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QPoint, QPointF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
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
    composition_mode,
    new_transparent_image,
    paint_stroke,
    stroke_hit,
)

TOOL_PEN = "pen"
TOOL_HIGHLIGHTER = "highlighter"
TOOL_ERASER = "eraser"
TOOL_AUTOMARK = "automark"  # 검출된 치수 박스를 클릭/드래그로 자동 형광 마킹

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
        ai = page.active_index

        below = new_transparent_image(size)
        bp = QPainter(below)
        bp.fillRect(below.rect(), Qt.GlobalColor.white)
        bp.drawImage(0, 0, page.base)
        for layer in page.layers[:ai]:
            if not layer.visible:
                continue
            bp.setOpacity(layer.opacity)
            bp.setCompositionMode(composition_mode(layer.blend))
            bp.drawImage(0, 0, layer.image)
        bp.end()
        self._below = below

        above = new_transparent_image(size)
        ap = QPainter(above)
        for layer in page.layers[ai + 1:]:
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
        painter.end()

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
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._panning = True
            self._pan_last = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            img_pt = self.view_to_image(QPointF(event.position()))
            if self.tools.tool == TOOL_AUTOMARK:
                self._automarking = True
                self.autoMarkRequested.emit(img_pt)  # 박스 채우기는 외부(메인윈도우)에서
            elif (self.tools.tool == TOOL_ERASER
                    and self.tools.eraser_mode == ERASE_MODE_STROKE):
                self._begin_erase(img_pt)   # 획 전체 지우기
            else:
                self._begin_stroke(img_pt)  # 펜/형광펜 또는 부분 지우기

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning and self._pan_last is not None:
            now = event.position().toPoint()
            self.offset += QPointF(now - self._pan_last)
            self._pan_last = now
            self.update()
            return
        img_pt = self.view_to_image(QPointF(event.position()))
        if self._drawing:
            self._continue_stroke(img_pt)
        elif self._erasing:
            self._erase_at(img_pt)
        elif self._automarking:
            self.autoMarkRequested.emit(img_pt)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._panning:
            self._panning = False
            self._pan_last = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if self._drawing:
            self._end_stroke()
        elif self._erasing:
            self._erasing = False
        elif self._automarking:
            self._automarking = False

    # ---------- 자동 마킹 / 화면 이동 ----------
    def mark_box_highlight(self, rect) -> bool:
        """주어진 이미지 좌표 사각형을 활성 레이어에 형광펜으로 채운다."""
        page = self.page
        if page is None or not page.active_layer.visible:
            return False
        self._push_undo()
        layer = page.active_layer
        c = self.tools.highlighter_color
        pad = 2
        yc = rect.center().y()
        pts = [(rect.left() - pad, yc), (rect.right() + pad, yc)]
        width = rect.height() + pad * 2
        stroke = Stroke(
            tool=TOOL_HIGHLIGHTER,
            color=(c.red(), c.green(), c.blue(), c.alpha()),
            width=float(width),
            opacity=self.tools.highlighter_opacity,
            points=pts,
        )
        layer.strokes.append(stroke)
        paint_stroke(layer.image, stroke)
        self.update()
        self.contentChanged.emit()
        return True

    def center_on_image_rect(self, rect):
        """이미지 좌표 사각형이 화면 중앙에 보이도록 줌·이동."""
        if rect.width() <= 0 or rect.height() <= 0:
            return
        margin = 0.35  # 사각형이 화면의 약 1/3 차지
        sx = self.width() * margin / rect.width()
        sy = self.height() * margin / rect.height()
        self.scale = max(MIN_SCALE, min(MAX_SCALE, min(sx, sy)))
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
