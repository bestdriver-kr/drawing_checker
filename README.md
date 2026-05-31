# Drawing Checker

도면/이미지(TIF·PNG 등) 위에 **펜·형광펜**으로 마킹하고, **레이어**별로 덧칠·관리하는 데스크톱 도구입니다. 멀티페이지 TIF(스캔 도면)를 지원합니다.

## 설치 / 실행

```powershell
pip install -r requirements.txt
python main.py
```

> ⚠️ 학습 모델 `models/drawing_crnn.pt`(약 30MB)는 저장소에 포함되지 않습니다(.gitignore).
> Drawing OCR 엔진을 쓰려면 모델 파일을 `models/` 폴더에 따로 넣어야 합니다.
> (모델이 없어도 다른 OCR 엔진과 모든 마킹/편집 기능은 정상 동작합니다.)

(현재 환경에는 PySide6·Pillow가 이미 설치되어 있습니다.)

## 사용법

1. **파일 → 이미지 열기**(Ctrl+O)로 **PDF**·TIF·PNG·JPG·BMP를 불러옵니다. PDF·멀티페이지 TIF는 여러 페이지로 들어오며 툴바의 페이지 이동(◀|/|▶)으로 넘깁니다. PDF는 200 DPI로 래스터화됩니다(이미지로 변환되어 위에 마킹).
2. 툴바에서 **도구**(펜 / 형광펜 / 지우개), **색상**, **굵기**를 고릅니다.
3. 캔버스에서 **마우스 왼쪽 드래그**로 그립니다.
   - **휠**: 커서 기준 확대/축소
   - **Ctrl+드래그** 또는 **휠 클릭 드래그**: 화면 이동(팬)
   - **Ctrl+0**: 화면 맞춤
4. 툴바의 **레이어** 드롭다운에서 활성 레이어를 고르고(맨 아래 `➕ 새 레이어 추가` 항목으로 추가), 옆의 버튼으로 이름변경(✎)·삭제(✕)·표시숨김(👁)·순서이동(▲▼)을 합니다. 둘째 줄의 **불투명도** 슬라이더와 **블렌드** 콤보로 활성 레이어의 속성을 조절합니다.
5. 저장/내보내기:
   - **파일 → 프로젝트 저장**(Ctrl+S) / **다른 이름으로 저장**(Ctrl+Shift+S): `.dck`(재편집용)와 `.pdf`(레이어 토글 공유용)를 **함께** 저장합니다. `.dck`는 **레이어·불투명도·블렌드·표시여부·순서·멀티페이지가 모두 보존**되어 다시 열면 레이어별로 편집할 수 있고, `.pdf`는 멀티페이지 한 파일에 OCG 레이어로 담겨 뷰어에서 켜고 끌 수 있습니다.
   - **파일 → 프로젝트 열기**(Ctrl+Shift+O): 저장한 `.dck`를 레이어 그대로 복원.
   - **파일 → TIFF 내보내기**: 보이는 레이어를 합쳐 TIFF로 저장(LZW 압축). 멀티페이지 문서는 한 파일에 묶을지 현재 페이지만 저장할지 물어봅니다.
   - **파일 → PDF 내보내기(레이어 토글)**: 각 레이어를 PDF의 OCG(Optional Content Group)로 저장. Acrobat 등 뷰어의 레이어 패널에서 켜고 끌 수 있습니다(배경 도면은 항상 표시, 레이어의 현재 표시여부가 기본 ON/OFF로 반영). 멀티페이지 지원.
   - **파일 → PNG 내보내기**(Ctrl+E): 보이는 레이어를 한 장으로 합쳐 저장(공유/인쇄용, 편집 불가).
   - **파일 → OpenRaster(.ora) 내보내기**: 현재 페이지를 표준 레이어 포맷으로 저장. GIMP·Krita에서 열립니다(단일 페이지).

## 파일 포맷

| 포맷 | 레이어 보존 | 멀티페이지 | 용도 |
|---|---|---|---|
| `.dck` (자체) | ✅ | ✅ | 작업 저장/재편집 |
| `.ora` (OpenRaster) | ✅ | ❌(페이지별) | 타 앱 호환 |
| `.pdf` (OCG) | ⚠️(토글만, 재편집 불가) | ✅ | 레이어 켜고 끄는 검토용 공유 |
| `.tif` (TIFF) | ❌(평탄화) | ✅ | 도면 표준, 멀티페이지 결과물 |
| `.png` | ❌(평탄화) | ❌ | 단일 이미지 공유 |

저장/내보내기는 모두 무료 라이선스 라이브러리만 사용합니다: Qt(PySide6, LGPL), Pillow(HPND), pikepdf(MPL, PDF용). 추가 비용·배포 제약이 없습니다.

## 마킹검사 (OCR)

도면에서 **텍스트(영문·숫자) 항목을 OCR로 찾아, 그 위에 마커(펜/형광펜)가 없는 항목을 경고**하는 검토 보조 기능입니다.

- 툴바/메뉴의 **마킹검사**(돋보기+체크 아이콘): 현재 페이지를 검사해 미마킹 항목을 **빨간 점선**으로 표시하고 목록을 보여줍니다.
- **저장/내보내기 시 자동 검사**: 미마킹 항목이 있으면 경고하고 계속/취소를 묻습니다. `검사 → 저장 시 마킹검사` 메뉴로 끌 수 있습니다.
- **마킹 판정**: 텍스트 영역에 보이는 주석 레이어 픽셀이 **조금이라도 닿으면** 마킹된 것으로 봅니다(숨긴 레이어는 미마킹으로 간주).
- **OCR 엔진 선택**(`검사 → OCR 엔진`):
  - **EasyOCR**: 딥러닝 기반, 정확도 높음. 첫 실행 시 모델 다운로드(~100MB), 페이지당 수초.
  - **Tesseract**: 가볍고 빠름(페이지당 0.x초). `tesseract.exe` 설치 필요(`winget install UB-Mannheim.TesseractOCR`). 미설치 시 선택하면 설치 안내가 뜹니다.
  - **RapidOCR**: ONNX 기반, 빠르고 설치 간단(모델 동봉). `pip install rapidocr_onnxruntime`.
  - **Windows OCR**: Windows 내장 OCR(winrt). 매우 빠르고 추가 설치 불필요. 설치된 OS 언어 팩(en/ko 등)을 사용합니다.
  - **Drawing OCR (도면 치수 전용)**: 동봉한 CRNN 모델. 숫자·공차기호(± Ø °)·ISO 286 끼워맞춤 코드(H7, g6) 등 **치수 텍스트에 특화**. 패키지 `drawing_ocr/`와 `models/drawing_crnn.pt`가 함께 들어 있어 별도 설치가 필요 없습니다(torch·opencv는 EasyOCR과 공유).
  - 엔진을 바꾸면 결과 캐시를 비우고 다시 검출합니다. 페이지·엔진별로 결과를 캐시합니다.

### OCR 엔진 직접 추가하기

다른 OCR(사내 모델, REST API, 다른 라이브러리 등)을 끼워 넣을 수 있습니다. [app/custom_ocr.py](app/custom_ocr.py)를 열어 예시 클래스를 본인 엔진으로 채우고 맨 아래 `register_engine(...)` 주석만 해제하면, `검사 → OCR 엔진` 메뉴에 자동으로 나타납니다.

엔진이 지켜야 할 계약(`app/ocr.py`의 `OcrEngine`):

```python
from app.ocr import OcrEngine, TextItem, register_engine, qimage_to_rgb
from PySide6.QtCore import QRect

class MyEngine(OcrEngine):
    key = "myengine"            # 고유 ID
    label = "내 OCR"            # 메뉴 표시 이름
    def is_available(self):     # 사용 가능 여부
        return True, ""
    def detect(self, base, langs, min_conf):
        rgb = qimage_to_rgb(base)          # (h,w,3) numpy, 원본 픽셀 좌표
        # ... 본인 엔진 호출 후 결과를 TextItem으로 변환 ...
        return [TextItem("ABC", QRect(x, y, w, h), conf)]   # rect=이미지 픽셀 좌표

register_engine(MyEngine())
```

핵심 규칙: `rect`는 반드시 **배경 이미지의 픽셀 좌표** QRect여야 마킹 판정과 화면 표시가 맞습니다.

## 도구 동작

- **펜**: 불투명 선. 색상·굵기 조절.
- **형광펜**: 반투명 선. 한 획 안에서 겹쳐도 진해지지 않도록 획 단위로 합성합니다. 레이어 블렌드를 **멀티플라이**로 두면 도면 선 위에서 형광펜 느낌이 더 자연스럽습니다.
- **지우개**: **획 단위**로 지웁니다. 클릭(또는 드래그로 스쳐 지나가면) 그 자리에 그려진 펜·형광펜 획을 **통째로** 제거합니다(부분 지우기 아님). 지우개 굵기는 클릭 인식 반경으로 쓰입니다. 배경 도면은 보존됩니다. 획 정보는 `.dck`에 함께 저장되어, 프로젝트를 다시 열어도 획 단위로 지울 수 있습니다.

## 구조

```
main.py                  진입점
app/image_loader.py      TIF/PNG는 Pillow, PDF는 pypdfium2로 → QImage
app/layer.py             Layer / Page / Document 모델
app/canvas.py            그리기 캔버스 (줌·팬·도구·합성·실행취소)
app/main_window.py       툴바·메뉴·레이어 컨트롤
app/project.py           .dck 저장/열기, .ora·.tif·.pdf 내보내기
app/ocr.py               OCR 텍스트 검출 + 미마킹 항목 판정 + 엔진 레지스트리
app/custom_ocr.py        사용자 엔진 등록(도면 치수 전용 drawing_ocr 어댑터 포함)
drawing_ocr/             도면 치수 CRNN OCR 패키지(동봉)
models/drawing_crnn.pt   학습된 CRNN 모델(동봉)
```

## 알려진 한계 / 향후

- `.ora`는 표준상 단일 이미지 스택이라 멀티페이지를 한 파일에 담지 못합니다(페이지별 내보내기).
- 실행취소는 획 단위 40단계.
- 매우 큰 도면에서 확대 시 부드러움보다 속도를 우선합니다.
- PDF의 OCG 레이어는 표시/숨김 토글만 지원하며, 불투명도·블렌드·획 재편집은 보존되지 않습니다(재편집은 `.dck` 사용).
