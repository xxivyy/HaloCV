import sys

from PySide6.QtWidgets import QApplication


def main() -> None:

    app = QApplication(sys.argv)

    from .tool_window import ToolWindow

    toolWindow = ToolWindow()
    toolWindow.show()

    exit_code = app.exec()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
