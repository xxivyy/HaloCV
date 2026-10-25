from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QCloseEvent, QMouseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
)

from .detection import Detector
from .overlay import DrawMode
from .tool_style import STYLE


class ToolWindow(QDialog):
    def __init__(self) -> None:
        super().__init__()

        self.detector: Detector | None = None

        self.setWindowTitle("HaloCV Settings")
        self.setFixedWidth(380)
        self.setStyleSheet(STYLE)
        self._drag_offset: QPoint | None = None

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
        )

        model_label = QLabel("MODEL")
        model_label.setObjectName("section")
        model_dir = Path("./.models")
        self.model_input = QComboBox()
        self.models = sorted(model_dir.glob("*.onnx"))
        for model in self.models:
            self.model_input.addItem(model.name, str(model))

        if not self.models:
            self.model_input.addItem("No ONNX models found")
            self.model_input.setEnabled(False)

        threshold_label = QLabel("DETECTION")
        threshold_label.setObjectName("section")
        self.confidence_input = self._threshold_input(0.15)
        self.iou_input = self._threshold_input(0.45)
        self.confidence_value = QLabel()
        self.confidence_value.setObjectName("value")
        self.iou_value = QLabel()
        self.iou_value.setObjectName("value")
        confidence_row = self._setting_row(
            "Confidence", self.confidence_input, self.confidence_value
        )
        iou_row = self._setting_row("IoU threshold", self.iou_input, self.iou_value)

        mode_label = QLabel("OVERLAY")
        mode_label.setObjectName("section")
        self.mode_group = QButtonGroup(self)
        self.mode_buttons: dict[DrawMode, QPushButton] = {}
        modes = QHBoxLayout()
        modes.setSpacing(6)
        for label, mode in (
            ("Boxes", DrawMode.BOXES),
            ("Pixelated", DrawMode.PIXELATE),
            ("Blur", DrawMode.BLUR),
        ):
            button = QPushButton(label)
            button.setObjectName("mode")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            modes.addWidget(button)
        self.mode_buttons[DrawMode.BOXES].setChecked(True)

        self.status_label = QLabel("Changes take effect immediately")
        self.status_label.setObjectName("status")
        average_label = QLabel("LAST SECOND")
        average_label.setObjectName("section")
        self.fps_value = QLabel("--.-")
        self.fps_value.setObjectName("fpsValue")
        fps_unit = QLabel("FPS")
        fps_unit.setObjectName("fpsUnit")
        fps_row = QHBoxLayout()
        fps_row.setSpacing(8)
        fps_row.addWidget(self.fps_value)
        fps_row.addWidget(fps_unit, alignment=Qt.AlignmentFlag.AlignBottom)
        fps_row.addStretch()

        eyebrow = QLabel("HALOCV  /  DESKTOP")
        eyebrow.setObjectName("eyebrow")

        title = QLabel("Settings")
        title.setObjectName("title")

        minimize_button = QToolButton()
        minimize_button.setObjectName("windowControl")
        minimize_button.setText("-")
        minimize_button.setToolTip("Minimize")
        minimize_button.setAccessibleName("Minimize window")
        minimize_button.setCursor(Qt.CursorShape.PointingHandCursor)
        minimize_button.clicked.connect(self.showMinimized)

        close_button = QToolButton()
        close_button.setObjectName("closeControl")
        close_button.setText("x")
        close_button.setToolTip("Exit")
        close_button.setAccessibleName("Close window")
        close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        close_button.clicked.connect(self.close)

        heading = QVBoxLayout()
        heading.setSpacing(3)
        heading.addWidget(eyebrow)
        heading.addWidget(title)

        window_controls = QHBoxLayout()
        window_controls.setSpacing(4)
        window_controls.addWidget(minimize_button)
        window_controls.addWidget(close_button)

        header = QHBoxLayout()
        header.setSpacing(12)
        header.addLayout(heading)
        header.addStretch()
        header.addLayout(window_controls)

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFrameShape(QFrame.Shape.HLine)

        settings_column = QVBoxLayout()
        settings_column.setSpacing(14)
        settings_column.addWidget(model_label)
        settings_column.addWidget(self.model_input)
        settings_column.addWidget(threshold_label)
        settings_column.addLayout(confidence_row)
        settings_column.addLayout(iou_row)
        settings_column.addWidget(mode_label)
        settings_column.addLayout(modes)
        settings_column.addWidget(average_label)
        settings_column.addLayout(fps_row)
        settings_column.addStretch(1)
        settings_column.addWidget(self.status_label)

        surface = QFrame(self)
        surface.setObjectName("surface")
        surface.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)

        layout = QVBoxLayout(surface)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)
        layout.addLayout(header)
        layout.addWidget(divider)
        layout.addLayout(settings_column, 1)

        window_layout = QVBoxLayout(self)
        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.addWidget(surface)

        self.confidence_input.valueChanged.connect(self._update_threshold_values)
        self.iou_input.valueChanged.connect(self._update_threshold_values)
        self.model_input.currentIndexChanged.connect(self._initialize_detector)
        self.mode_group.buttonClicked.connect(self._emit_settings)
        self._emit_settings()
        self._update_threshold_values()
        self._initialize_detector()

    def _initialize_detector(self, *_args) -> None:
        self._stop_detector()

        model_path = self.model_input.currentData()

        if model_path is None or not self.models:
            return

        self.detector = Detector(
            model_path=str(model_path),
            conf=self.confidence_input.value(),
            iou=self.iou_input.value(),
        )

        self.detector.fps_signal.connect(self._update_fps)

    def _stop_detector(self) -> None:
        if self.detector is not None:
            self.detector.stop()
            self.detector = None

    def _emit_settings(self, *_args) -> None:
        Detector.overlay.set_draw_mode(
            next(
                mode for mode, button in self.mode_buttons.items() if button.isChecked()
            )
        )

    @staticmethod
    def _threshold_input(value: float) -> QDoubleSpinBox:
        spin_box = QDoubleSpinBox()
        spin_box.setRange(0.0, 1.0)
        spin_box.setDecimals(2)
        spin_box.setSingleStep(0.05)
        spin_box.setValue(value)
        return spin_box

    @staticmethod
    def _setting_row(label: str, control, value: QLabel) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        field_label = QLabel(label)
        field_label.setObjectName("fieldLabel")
        row.addWidget(field_label)
        row.addStretch()
        row.addWidget(value)
        row.addWidget(control)
        return row

    def _update_threshold_values(self, *_args) -> None:
        self.confidence_value.setText(f"{self.confidence_input.value():.2f}")
        self.iou_value.setText(f"{self.iou_input.value():.2f}")

        if self.detector is not None:
            self.detector.conf = self.confidence_input.value()
            self.detector.iou = self.iou_input.value()

    def _update_fps(self, fps: int) -> None:
        self.fps_value.setText(f"{fps}")

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop_detector()
        super().closeEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() <= 58:
            self._drag_offset = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._drag_offset is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = None
        super().mouseReleaseEvent(event)
