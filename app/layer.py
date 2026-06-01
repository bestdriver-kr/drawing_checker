"""레이어 및 문서(페이지 모음) 모델."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen

BLEND_NORMAL = "normal"
BLEND_MULTIPLY = "multiply"

# 획 종류(캔버스의 TOOL_* 문자열과 동일 값)
STROKE_PEN = "pen"
STROKE_HIGHLIGHTER = "highlighter"
STROKE_ERASER = "eraser"  # 부분 지우개 획(픽셀을 투명하게 지움)
STROKE_AUTOMARK = "automark_mask"  # 자동마킹: OCR 글자 픽셀만 칠하는 마스크 획
STROKE_TEXT = "text"  # 텍스트 입력(글자를 레이어에 그림)

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


def _resized_canvas(src: QImage, w: int, h: int, white: bool) -> QImage:
    """src를 새 (w, h) 캔버스에 좌상단 기준으로 옮긴다(배경=흰색/투명)."""
    if white:
        fmt = src.format()
        if fmt == QImage.Format.Format_Invalid:
            fmt = QImage.Format.Format_RGBA8888
        out = QImage(w, h, fmt)
        out.fill(Qt.GlobalColor.white)
    else:
        out = new_transparent_image(QSize(w, h))
    p = QPainter(out)
    p.drawImage(0, 0, src)
    p.end()
    return out


@dataclass
class Stroke:
    """펜/형광펜으로 그린 한 획(객체 단위로 지우기 위해 보존)."""

    tool: str                       # STROKE_PEN | STROKE_HIGHLIGHTER | STROKE_AUTOMARK
    color: tuple                    # (r, g, b, a)
    width: float
    opacity: float                  # 형광펜 합성 불투명도(펜=1.0)
    points: list = field(default_factory=list)  # [(x, y), ...] 이미지 좌표
    # 자동마킹(STROKE_AUTOMARK) 전용: 글자 픽셀 알파 마스크(PNG bytes).
    # points = [(좌상 x, y), (우하 x, y)] 로 마스크 배치 위치를 정한다.
    mask: bytes | None = None
    # 텍스트(STROKE_TEXT) 전용: 글자 내용. width=글자 픽셀 크기, points[0]=좌상단.
    text: str = ""
    angle: float = 0.0  # 텍스트 회전 각도(도, 시계방향 +)


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


def _alpha_to_png(mask: "np.ndarray") -> bytes:
    """(h, w) uint8 마스크를 8비트 그레이 PNG 바이트로 인코딩."""
    h, w = mask.shape
    m = np.ascontiguousarray(mask, dtype=np.uint8)
    img = QImage(m.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(ba.data())


def _png_to_alpha(data: bytes) -> "np.ndarray":
    """_alpha_to_png 로 만든 PNG를 (h, w) uint8 마스크로 디코딩."""
    img = QImage()
    img.loadFromData(data, "PNG")
    img = img.convertToFormat(QImage.Format.Format_Grayscale8)
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)
    return arr[:, :w].copy()


def _dilate(mask: "np.ndarray", r: int) -> "np.ndarray":
    """4방향 팽창 r회(글자 마스크를 살짝 두껍게 → 마킹 가독성)."""
    out = mask
    for _ in range(max(0, int(r))):
        m = out
        out = m.copy()
        out[1:, :] = np.maximum(out[1:, :], m[:-1, :])
        out[:-1, :] = np.maximum(out[:-1, :], m[1:, :])
        out[:, 1:] = np.maximum(out[:, 1:], m[:, :-1])
        out[:, :-1] = np.maximum(out[:, :-1], m[:, 1:])
    return out


def build_automark_stroke(base: QImage, rect: QRect, color_rgba: tuple,
                          opacity: float, grow: int) -> Stroke | None:
    """OCR 박스 내부의 '글자(어두운) 픽셀'만 골라 칠하는 자동마킹 획을 만든다.

    박스 전체를 채우지 않고 인식된 글자 모양만 따라 칠하므로 주변 치수선/
    여백을 건드리지 않는다. grow로 글자를 살짝 두껍게 해 가독성을 높인다.
    """
    r = rect.intersected(base.rect())
    if r.isEmpty():
        return None
    sub = base.copy(r).convertToFormat(QImage.Format.Format_Grayscale8)
    w, h = sub.width(), sub.height()
    if w <= 0 or h <= 0:
        return None
    bpl = sub.bytesPerLine()
    arr = np.frombuffer(sub.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)[:, :w]
    thr = max(40, int(int(arr.max()) * 0.6))  # 밝은 배경 위 어두운 글자
    mask = np.where(arr < thr, np.uint8(255), np.uint8(0))
    mask = _dilate(mask, grow)
    if not mask.any():  # 글자 픽셀을 못 찾으면 박스 중앙에 얇은 띠로 대체
        yc, half = h // 2, max(1, h // 5)
        mask[max(0, yc - half):yc + half, :] = 255
    return Stroke(
        tool=STROKE_AUTOMARK,
        color=tuple(color_rgba),
        width=float(h),
        opacity=opacity,
        points=[(r.left(), r.top()), (r.right(), r.bottom())],
        mask=_alpha_to_png(mask),
    )


def _paint_automark(image: QImage, stroke: Stroke):
    if not stroke.mask or len(stroke.points) < 1:
        return
    x0, y0 = stroke.points[0]
    mask = _png_to_alpha(stroke.mask)
    h, w = mask.shape
    rr, gg, bb = stroke.color[0], stroke.color[1], stroke.color[2]
    af = mask.astype(np.uint16)
    buf = np.zeros((h, w, 4), np.uint8)  # 메모리상 BGRA(ARGB32 리틀엔디안)
    buf[..., 0] = (bb * af // 255).astype(np.uint8)
    buf[..., 1] = (gg * af // 255).astype(np.uint8)
    buf[..., 2] = (rr * af // 255).astype(np.uint8)
    buf[..., 3] = mask
    buf = np.ascontiguousarray(buf)
    qbuf = QImage(buf.data, w, h, 4 * w,
                  QImage.Format.Format_ARGB32_Premultiplied).copy()
    p = QPainter(image)
    p.setOpacity(stroke.opacity)
    p.drawImage(int(x0), int(y0), qbuf)
    p.end()


def _text_font(stroke: Stroke) -> QFont:
    font = QFont()
    font.setPixelSize(max(4, int(round(stroke.width))))
    return font


def text_bounds(stroke: Stroke) -> tuple[int, int]:
    """텍스트 획의 (폭, 높이) 픽셀 크기."""
    fm = QFontMetrics(_text_font(stroke))
    lines = stroke.text.split("\n") or [""]
    w = max((fm.horizontalAdvance(ln) for ln in lines), default=0)
    h = fm.height() * max(1, len(lines))
    return w, h


def _paint_text(image: QImage, stroke: Stroke):
    if not stroke.text or not stroke.points:
        return
    x, y = stroke.points[0]
    font = _text_font(stroke)
    fm = QFontMetrics(font)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    p.setFont(font)
    p.setPen(QColor(*stroke.color))
    lh, asc = fm.height(), fm.ascent()
    p.translate(x, y)              # 좌상단을 기준점으로
    if stroke.angle:
        p.rotate(stroke.angle)     # 시계방향 +
    for i, line in enumerate(stroke.text.split("\n")):
        p.drawText(0, asc + i * lh, line)
    p.end()


def paint_stroke(image: QImage, stroke: Stroke):
    """한 획을 이미지에 그린다. 형광펜은 획 단위 버퍼로 자기겹침 누적을 막는다."""
    if stroke.tool == STROKE_AUTOMARK:
        _paint_automark(image, stroke)
        return
    if stroke.tool == STROKE_TEXT:
        _paint_text(image, stroke)
        return
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
    if stroke.tool == STROKE_AUTOMARK and len(stroke.points) >= 2:
        (x0, y0), (x1, y1) = stroke.points[0], stroke.points[1]
        return (x0 - tol) <= px <= (x1 + tol) and (y0 - tol) <= py <= (y1 + tol)
    if stroke.tool == STROKE_TEXT and stroke.points:
        x, y = stroke.points[0]
        w, h = text_bounds(stroke)
        return (x - tol) <= px <= (x + w + tol) and (y - tol) <= py <= (y + h + tol)
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

    def resize_canvas(self, w: int, h: int):
        """캔버스(페이지) 크기를 바꾼다. 좌상단 고정 — 키우면 흰/투명으로 채우고,
        줄이면 잘라낸다. 그림은 좌상단 기준으로 보존된다."""
        w, h = max(1, int(w)), max(1, int(h))
        self.base = _resized_canvas(self.base, w, h, white=True)
        for layer in self.layers:
            layer.image = _resized_canvas(layer.image, w, h, white=False)

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
