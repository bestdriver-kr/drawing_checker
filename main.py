"""Drawing Checker 진입점."""
import sys

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow, make_app_icon


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
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
