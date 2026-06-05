"""프로젝트 저장/열기.

- `.dck`: 자체 ZIP 포맷. 멀티페이지 + 레이어(이름/불투명도/블렌드/표시/순서) 완전 보존.
- `.ora`: OpenRaster 표준(단일 페이지). GIMP/Krita 등에서 열림.

모두 Python 표준 라이브러리(zipfile/json)와 Qt PNG 인코딩만 사용 → 추가 의존성/라이선스 비용 없음.
"""
from __future__ import annotations

import base64
import json
import zlib
import zipfile
from io import BytesIO
from xml.sax.saxutils import escape

from PIL import Image
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtGui import QImage, QPainter

from .layer import (
    BLEND_MULTIPLY,
    BLEND_NORMAL,
    MODE_PROGRAM,
    Document,
    Layer,
    Page,
    Stroke,
    composition_mode,
)

DCK_VERSION = 1

# 우리 블렌드 ↔ OpenRaster composite-op 매핑
_BLEND_TO_ORA = {
    BLEND_NORMAL: "svg:src-over",
    BLEND_MULTIPLY: "svg:multiply",
}
_ORA_TO_BLEND = {v: k for k, v in _BLEND_TO_ORA.items()}


def _qimage_to_png_bytes(img: QImage) -> bytes:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(ba)


def _png_bytes_to_qimage(data: bytes) -> QImage:
    img = QImage()
    img.loadFromData(data, "PNG")
    return img


def _as_premultiplied(img: QImage) -> QImage:
    return img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)


def _qimage_to_pil(img: QImage) -> Image.Image:
    """QImage를 PIL 이미지(RGB)로 변환(PNG 경유로 stride 문제 회피)."""
    return Image.open(BytesIO(_qimage_to_png_bytes(img))).convert("RGB")


def export_tiff(flats: list[QImage], path: str) -> None:
    """평탄화된 페이지 이미지들을 (멀티)페이지 TIFF로 저장한다(LZW 무손실 압축)."""
    pil_imgs = [_qimage_to_pil(f) for f in flats]
    first, rest = pil_imgs[0], pil_imgs[1:]
    first.save(
        path,
        format="TIFF",
        save_all=True,
        append_images=rest,
        compression="tiff_lzw",
    )


def _qimage_rgb_alpha(img: QImage) -> tuple[int, int, bytes, bytes]:
    """QImage → (w, h, RGB bytes, alpha bytes). PNG 경유로 stride 안전하게 추출."""
    pil = Image.open(BytesIO(_qimage_to_png_bytes(img))).convert("RGBA")
    w, h = pil.size
    rgb = pil.convert("RGB").tobytes()
    alpha = pil.getchannel("A").tobytes()
    return w, h, rgb, alpha


def export_pdf(pages: list[Page], path: str) -> None:
    """페이지들을 OCG(레이어 토글) PDF로 내보낸다.

    배경 도면은 항상 표시, 주석 레이어는 각각 켜고 끌 수 있는 OCG가 된다.
    레이어의 현재 표시여부가 PDF 기본 ON/OFF 상태로 반영된다.
    """
    import pikepdf
    from pikepdf import Array, Dictionary, Name, Stream

    pdf = pikepdf.Pdf.new()
    all_ocgs = []
    order = []
    on_list = []
    off_list = []

    def make_image(w, h, rgb, alpha):
        smask = Stream(pdf, zlib.compress(alpha))
        smask.Type = Name.XObject
        smask.Subtype = Name.Image
        smask.Width = w
        smask.Height = h
        smask.ColorSpace = Name.DeviceGray
        smask.BitsPerComponent = 8
        smask.Filter = Name.FlateDecode
        im = Stream(pdf, zlib.compress(rgb))
        im.Type = Name.XObject
        im.Subtype = Name.Image
        im.Width = w
        im.Height = h
        im.ColorSpace = Name.DeviceRGB
        im.BitsPerComponent = 8
        im.Filter = Name.FlateDecode
        im.SMask = pdf.make_indirect(smask)
        return pdf.make_indirect(im)

    multipage = len(pages) > 1
    for pi, page in enumerate(pages):
        w, h = page.size.width(), page.size.height()
        page_obj = pdf.add_blank_page(page_size=(w, h))

        xobjects = Dictionary()
        properties = Dictionary()
        content = [b"q", f"{w} 0 0 {h} 0 0 cm".encode(), b"/ImBase Do", b"Q"]

        bw, bh, brgb, ba = _qimage_rgb_alpha(page.base)
        xobjects[Name("/ImBase")] = make_image(bw, bh, brgb, ba)

        # 현재 모드 레이어만, 상단이 PDF 패널 위에 오도록 역순 처리
        mlayers = _mode_layers(page)
        for li in range(len(mlayers) - 1, -1, -1):
            layer = mlayers[li]
            iw, ih, irgb, ia = _qimage_rgb_alpha(layer.image)
            im_name = f"Im{li}"
            oc_name = f"OC{li}"
            xobjects[Name("/" + im_name)] = make_image(iw, ih, irgb, ia)

            label = layer.name + (f" (p{pi + 1})" if multipage else "")
            ocg = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=label))
            properties[Name("/" + oc_name)] = ocg
            all_ocgs.append(ocg)
            order.append(ocg)
            (on_list if layer.visible else off_list).append(ocg)

            content += [
                f"/OC /{oc_name} BDC".encode(),
                b"q",
                f"{w} 0 0 {h} 0 0 cm".encode(),
                f"/{im_name} Do".encode(),
                b"Q",
                b"EMC",
            ]

        stream = Stream(pdf, b"\n".join(content))
        page_obj.Contents = pdf.make_indirect(stream)
        page_obj.Resources = Dictionary(XObject=xobjects, Properties=properties)

    pdf.Root.OCProperties = Dictionary(
        OCGs=Array(all_ocgs),
        D=Dictionary(
            Order=Array(order),
            ON=Array(on_list),
            OFF=Array(off_list),
            BaseState=Name.ON,
        ),
    )
    pdf.save(path)


def _mode_layers(page: Page) -> list:
    """현재 모드(레이어 그룹)에 속한 레이어만 반환(모드 개념 없으면 전체)."""
    mode = getattr(page, "current_mode", None)
    if mode is None:
        return list(page.layers)
    return [l for l in page.layers if getattr(l, "group", mode) == mode]


def flatten_page(page: Page) -> QImage:
    """한 페이지의 배경 + (현재 모드의) 보이는 레이어를 한 장으로 합친다(흰 배경)."""
    flat = QImage(page.size, QImage.Format.Format_ARGB32_Premultiplied)
    flat.fill(Qt.GlobalColor.white)
    p = QPainter(flat)
    p.drawImage(0, 0, page.base)
    for layer in _mode_layers(page):
        if not layer.visible:
            continue
        p.setOpacity(layer.opacity)
        p.setCompositionMode(composition_mode(layer.blend))
        p.drawImage(0, 0, layer.image)
    p.end()
    return flat


# ---------------------------------------------------------------- .dck
def save_dck(doc: Document, path: str) -> None:
    """문서를 .dck(ZIP)로 저장한다."""
    manifest = {
        "version": DCK_VERSION,
        "current_index": doc.current_index,
        "pages": [],
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for pi, page in enumerate(doc.pages):
            base_path = f"pages/{pi}/base.png"
            zf.writestr(base_path, _qimage_to_png_bytes(page.base))
            page_meta = {
                "size": [page.size.width(), page.size.height()],
                "active_index": page.active_index,
                "current_mode": getattr(page, "current_mode", None),
                "base": base_path,
                "layers": [],
            }
            for li, layer in enumerate(page.layers):
                layer_path = f"pages/{pi}/layers/{li}.png"
                zf.writestr(layer_path, _qimage_to_png_bytes(layer.image))
                page_meta["layers"].append({
                    "name": layer.name,
                    "file": layer_path,
                    "visible": layer.visible,
                    "opacity": layer.opacity,
                    "blend": layer.blend,
                    "group": getattr(layer, "group", None),
                    "stamp_name": getattr(layer, "stamp_name", ""),
                    "strokes": [{
                        "tool": s.tool,
                        "color": list(s.color),
                        "width": s.width,
                        "opacity": s.opacity,
                        "points": [[x, y] for x, y in s.points],
                        **({"mask": base64.b64encode(s.mask).decode("ascii")}
                           if s.mask else {}),
                        **({"text": s.text} if s.text else {}),
                        **({"angle": s.angle} if s.angle else {}),
                        **({"boxed": True} if s.boxed else {}),
                    } for s in layer.strokes],
                })
            manifest["pages"].append(page_meta)
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))


def save_bundle(doc: Document, dck_path: str) -> list[str]:
    """기본 저장: .dck(재편집용)와 .pdf(레이어 토글 공유용)를 함께 저장.

    저장된 경로 목록을 반환한다.
    """
    save_dck(doc, dck_path)
    written = [dck_path]

    stem = dck_path[:-4] if dck_path.lower().endswith(".dck") else dck_path
    pdf_path = f"{stem}.pdf"
    export_pdf(doc.pages, pdf_path)  # 멀티페이지 한 파일 + OCG 레이어
    written.append(pdf_path)
    return written


def load_dck(path: str) -> Document:
    """.dck를 읽어 Document로 복원한다."""
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

        pages: list[Page] = []
        for page_meta in manifest["pages"]:
            base = _png_bytes_to_qimage(zf.read(page_meta["base"]))
            page = Page(base)  # 기본 레이어 1장이 생기지만 곧 교체
            layers: list[Layer] = []
            for lm in page_meta["layers"]:
                img = _as_premultiplied(_png_bytes_to_qimage(zf.read(lm["file"])))
                layer = Layer(lm["name"], base.size(), image=img,
                              group=lm.get("group") or MODE_PROGRAM)
                layer.visible = bool(lm.get("visible", True))
                layer.opacity = float(lm.get("opacity", 1.0))
                layer.blend = lm.get("blend", BLEND_NORMAL)
                layer.stamp_name = lm.get("stamp_name", "") or ""
                layer.strokes = [
                    Stroke(
                        tool=sd["tool"],
                        color=tuple(sd["color"]),
                        width=sd["width"],
                        opacity=sd["opacity"],
                        points=[(px, py) for px, py in sd["points"]],
                        mask=(base64.b64decode(sd["mask"]) if sd.get("mask") else None),
                        text=sd.get("text", ""),
                        angle=float(sd.get("angle", 0.0)),
                        boxed=bool(sd.get("boxed", False)),
                    )
                    for sd in lm.get("strokes", [])
                ]
                layers.append(layer)
            if layers:
                page.layers = layers
            # 모드 복원(구버전 .dck는 전부 프로그램 그룹)
            page.current_mode = page_meta.get("current_mode") or MODE_PROGRAM
            if not page.mode_indices(page.current_mode):
                page.current_mode = page.layers[0].group
            ai = min(page_meta.get("active_index", 0), len(page.layers) - 1)
            # 활성 레이어는 현재 모드 그룹 안에 있어야 함
            if page.layers[ai].group != page.current_mode:
                ai = page.mode_indices(page.current_mode)[0]
            page.active_index = ai
            pages.append(page)

    # Document를 만들고 페이지를 복원본으로 교체
    doc = Document([p.base for p in pages], path)
    doc.pages = pages
    doc.current_index = min(manifest.get("current_index", 0), len(pages) - 1)
    return doc


# ---------------------------------------------------------------- .ora
def _ora_stack_xml(page: Page) -> str:
    w, h = page.size.width(), page.size.height()
    lines = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        f'<image w="{w}" h="{h}" version="0.0.3">',
        "  <stack>",
    ]
    # 현재 모드 레이어만. 최상단이 먼저 오도록 역순.
    mlayers = _mode_layers(page)
    n = len(mlayers)
    for li in range(n - 1, -1, -1):
        layer = mlayers[li]
        comp = _BLEND_TO_ORA.get(layer.blend, "svg:src-over")
        vis = "visible" if layer.visible else "hidden"
        lines.append(
            f'    <layer name="{escape(layer.name)}" src="data/layer{li}.png" '
            f'opacity="{layer.opacity:.3f}" visibility="{vis}" '
            f'composite-op="{comp}" x="0" y="0"/>'
        )
    # 배경 도면을 맨 아래 레이어로
    lines.append(
        '    <layer name="배경(도면)" src="data/base.png" opacity="1.000" '
        'visibility="visible" composite-op="svg:src-over" x="0" y="0"/>'
    )
    lines += ["  </stack>", "</image>"]
    return "\n".join(lines)


def export_ora(page: Page, merged: QImage, path: str) -> None:
    """한 페이지를 OpenRaster(.ora)로 내보낸다. merged는 합쳐진 미리보기 이미지."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        # mimetype은 반드시 첫 항목 + 무압축(스펙)
        mt = zipfile.ZipInfo("mimetype")
        mt.compress_type = zipfile.ZIP_STORED
        zf.writestr(mt, "image/openraster")

        zf.writestr("stack.xml", _ora_stack_xml(page))
        zf.writestr("data/base.png", _qimage_to_png_bytes(page.base))
        for li, layer in enumerate(_mode_layers(page)):
            zf.writestr(f"data/layer{li}.png", _qimage_to_png_bytes(layer.image))
        zf.writestr("mergedimage.png", _qimage_to_png_bytes(merged))
        thumb = merged.scaled(
            256, 256,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        zf.writestr("Thumbnails/thumbnail.png", _qimage_to_png_bytes(thumb))
