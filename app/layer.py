"""레이어 및 문서(페이지 모음) 모델."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRect, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen

BLEND_NORMAL = "normal"
BLEND_MULTIPLY = "multiply"

# 획 종류(캔버스의 TOOL_* 문자열과 동일 값)
STROKE_PEN = "pen"
STROKE_HIGHLIGHTER = "highlighter"
STROKE_ERASER = "eraser"  # 부분 지우개 획(픽셀을 투명하게 지움)

_BLEND_TO_MODE = {
    BLEND_NORMAL: QPainter.CompositionMode.CompositionMode_SourceOver,
    BLEND_MULTIPLY: QPainter.CompositionMode.CompositionMode_Multiply,
}


def composition_mode(blend: str) -> QPainter.CompositionMode:
    return _BLEND_TO_MODE.get(blend, QPainter.CompositionMode.CompositionMode_SourceOver)


def new_transparent_image(size: QSize) -> QImage:
    img = QImage(size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    return img


@dataclass
class Stroke:
    """펜/형광펜으로 그린 한 획(객체 단위로 지우기 위해 보존)."""

    tool: str                       # STROKE_PEN | STROKE_HIGHLIGHTER
    color: tuple                    # (r, g, b, a)
    width: float
    opacity: float                  # 형광펜 합성 불투명도(펜=1.0)
    points: list = field(default_factory=list)  # [(x, y), ...] 이미지 좌표


def _stroke_qpen(color_rgba, width: float) -> QPen:
    pen = QPen(QColor(*color_rgba), width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _draw_polyline(painter: QPainter, pts: list[QPointF]):
    if len(pts) == 1:
        painter.drawPoint(pts[0])
    else:
        painter.drawPolyline(pts)


def paint_stroke(image: QImage, stroke: Stroke):
    """한 획을 이미지에 그린다. 형광펜은 획 단위 버퍼로 자기겹침 누적을 막는다."""
    pts = [QPointF(x, y) for x, y in stroke.points]
    if not pts:
        return
    if stroke.tool == STROKE_ERASER:  # 부분 지우개: 픽셀을 투명하게
        p = QPainter(image)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        p.setPen(_stroke_qpen((0, 0, 0, 255), stroke.width))
        _draw_polyline(p, pts)
        p.end()
    elif stroke.tool == STROKE_HIGHLIGHTER:
        margin = int(stroke.width / 2 + 2)
        xs = [p.x() for p in pts]
        ys = [p.y() for p in pts]
        rect = QRect(
            int(min(xs)) - margin, int(min(ys)) - margin,
            int(max(xs) - min(xs)) + 2 * margin,
            int(max(ys) - min(ys)) + 2 * margin,
        ).intersected(image.rect())
        if rect.isEmpty():
            return
        buf = new_transparent_image(rect.size())
        bp = QPainter(buf)
        bp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bp.translate(-rect.topLeft())
        color = list(stroke.color)
        color[3] = 255  # 버퍼엔 불투명, 합성 시 opacity 적용
        bp.setPen(_stroke_qpen(color, stroke.width))
        _draw_polyline(bp, pts)
        bp.end()
        p = QPainter(image)
        p.setOpacity(stroke.opacity)
        p.drawImage(rect.topLeft(), buf)
        p.end()
    else:  # 펜
        p = QPainter(image)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(_stroke_qpen(stroke.color, stroke.width))
        _draw_polyline(p, pts)
        p.end()


def _dist_point_segment(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def stroke_hit(stroke: Stroke, px: float, py: float, tol: float) -> bool:
    """점 (px, py)가 획에서 (선폭/2 + tol) 이내면 True."""
    r = stroke.width / 2 + tol
    pts = stroke.points
    if len(pts) == 1:
        return math.hypot(px - pts[0][0], py - pts[0][1]) <= r
    for i in range(len(pts) - 1):
        if _dist_point_segment(px, py, pts[i][0], pts[i][1],
                                pts[i + 1][0], pts[i + 1][1]) <= r:
            return True
    return False


class Layer:
    """그림을 그릴 수 있는 투명 레이어 한 장."""

    def __init__(self, name: str, size: QSize, image: QImage | None = None):
        self.name = name
        self.image = image if image is not None else new_transparent_image(size)
        self.visible = True
        self.opacity = 1.0
        self.blend = BLEND_NORMAL
        self.strokes: list[Stroke] = []  # 객체 단위 지우개용 획 기록

    def rerender(self):
        """보존된 획들로부터 이미지를 다시 그린다(획 삭제 후 호출)."""
        self.image = new_transparent_image(self.image.size())
        for stroke in self.strokes:
            paint_stroke(self.image, stroke)


# 새 문서를 열 때 기본으로 만들어지는 레이어 이름(아래→위 순서)
DEFAULT_LAYER_NAMES = ["CNC/MTM", "MCT"]


class Page:
    """배경 이미지 한 장과 그 위의 주석 레이어들."""

    def __init__(self, base_image: QImage):
        self.base = base_image
        size = base_image.size()
        self.layers: list[Layer] = [Layer(name, size) for name in DEFAULT_LAYER_NAMES]
        self.active_index = 0

    @property
    def size(self) -> QSize:
        return self.base.size()

    @property
    def active_layer(self) -> Layer:
        return self.layers[self.active_index]

    def add_layer(self, name: str | None = None) -> Layer:
        name = name or f"레이어 {len(self.layers) + 1}"
        layer = Layer(name, self.size)
        # 활성 레이어 바로 위에 삽입
        insert_at = self.active_index + 1
        self.layers.insert(insert_at, layer)
        self.active_index = insert_at
        return layer

    def remove_active_layer(self) -> bool:
        if len(self.layers) <= 1:
            return False  # 최소 한 장은 유지
        del self.layers[self.active_index]
        self.active_index = min(self.active_index, len(self.layers) - 1)
        return True

    def move_active(self, delta: int) -> bool:
        new_index = self.active_index + delta
        if not (0 <= new_index < len(self.layers)):
            return False
        layers = self.layers
        layers[self.active_index], layers[new_index] = (
            layers[new_index],
            layers[self.active_index],
        )
        self.active_index = new_index
        return True


class Document:
    """열린 파일 하나에 대응하는 페이지 모음."""

    def __init__(self, pages: list[QImage], path: str | None = None):
        self.path = path
        self.pages: list[Page] = [Page(img) for img in pages]
        self.current_index = 0

    @property
    def current_page(self) -> Page:
        return self.pages[self.current_index]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def set_page(self, index: int) -> bool:
        if 0 <= index < len(self.pages):
            self.current_index = index
            return True
        return False
