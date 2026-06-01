"""Drawing Checker 진입점."""
import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow, make_app_icon


def _file_arg(argv) -> str | None:
    """명령행 인자에서 열 파일 경로를 찾는다(더블클릭/파일 연결 시 OS가 넘김)."""
    for a in argv[1:]:
        if a and not a.startswith("-") and os.path.isfile(a):
            return os.path.abspath(a)
    return None


def _set_windows_app_id():
    """윈도우 작업표시줄이 python.exe 대신 이 앱의 아이콘을 쓰도록 AppUserModelID 지정."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "DrawingChecker.App"
        )
    except Exception:  # noqa: BLE001
        pass


def main():
    _set_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName("Drawing Checker")
    app.setWindowIcon(make_app_icon())

    # 글자 크기 약 25% 확대
    font = app.font()
    if font.pointSizeF() > 0:
        font.setPointSizeF(font.pointSizeF() * 1.25)
    else:
        font.setPixelSize(int(font.pixelSize() * 1.25))
    app.setFont(font)
    window = MainWindow()
    window.showMaximized()  # 크게 띄워 하얀 캔버스가 작업영역을 채우도록
    path = _file_arg(sys.argv)  # 더블클릭/파일 연결로 넘어온 파일 즉시 열기
    if path:
        window._open_path(path)
    else:
        # 레이아웃이 잡힌(최대화 적용) 뒤 캔버스 크기에 맞춰 흰 바탕 생성
        QTimer.singleShot(0, window.new_blank)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
