# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 빌드 스펙 — Drawing Checker (torch/Drawing OCR/RapidOCR 등 포함)
# 빌드:  pyinstaller build_exe.spec
# 결과:  dist/DrawingChecker/DrawingChecker.exe  (onedir, 수 GB)
import glob
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [("app_icon.ico", ".")]
binaries = []
hiddenimports = []

# Drawing OCR v2 모델 동봉: TrOCR 파인튜닝 + CRNN + CRAFT_Lite
for _f in glob.glob(os.path.join("models", "trocr_finetuned", "*")):
    if os.path.isfile(_f):
        datas.append((_f, "models/trocr_finetuned"))
for _m in ("craft_lite.pt", "drawing_crnn.pt"):
    if os.path.isfile(os.path.join("models", _m)):
        datas.append((os.path.join("models", _m), "models"))

# 무거운 의존성을 통째로 수집(설치 안 된 건 건너뜀)
for pkg in [
    "torch", "torchvision", "transformers", "tokenizers", "safetensors",
    "sentencepiece", "onnxruntime", "rapidocr_onnxruntime",
    "cv2", "scipy", "skimage", "pikepdf", "pypdfium2", "shapely",
    "PIL", "numpy", "yaml",
]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

hiddenimports += collect_submodules("drawing_ocr_v2")
hiddenimports += collect_submodules("app")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DrawingChecker",
    console=False,
    icon="app_icon.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="DrawingChecker",
)
