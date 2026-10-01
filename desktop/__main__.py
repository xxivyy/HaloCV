import sys

from PySide6.QtWidgets import QApplication


def main() -> None:

    app = QApplication(sys.argv)

    from .overlay import Overlay

    overlay = Overlay()

    from .screen import Screen

    Screen.capture(overlay)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
