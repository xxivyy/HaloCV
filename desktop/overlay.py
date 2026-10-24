from enum import Enum, auto

import cv2
import numpy as np
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPalette, QRegion
from PySide6.QtWidgets import QWidget

from .winapi import WinAPI


class DrawMode(Enum):
    BOX = auto()
    BLUR = auto()
    PIXELATE = auto()


class Overlay(QWidget):
    draw_signal = Signal(list, np.ndarray)  # detections, frame

    def __init__(self) -> None:
        super().__init__()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )

        self.transparency_rgb = (1, 2, 3)
        hwnd = int(self.winId())

        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(*self.transparency_rgb))
        self.setPalette(palette)

        WinAPI.set_transparency(hwnd, self.transparency_rgb)
        WinAPI.exclude_from_capture(hwnd)

        self.showFullScreen()

        self.detections: list[dict] = []
        self.frame: np.ndarray = np.empty((1, 1, 3), dtype=np.uint8)

        self.draw_signal.connect(self._draw_event)
        self.draw_region = QRegion()
        self.draw_mode = DrawMode.PIXELATE

        self.pixelate_block_size = 25

    def _get_draw_region(self, detections: list[dict]) -> QRegion:
        region = QRegion()

        for d in detections:
            rect = QRect(*d["box"]).adjusted(-3, -3, 3, 3)
            region = region.united(QRegion(rect))

        return region

    def _draw_event(self, detections: list[dict], frame: np.ndarray) -> None:
        old_region = self.draw_region
        new_region = self._get_draw_region(detections)

        self.detections = detections
        self.frame = frame

        self.draw_region = new_region
        self.update(old_region.united(new_region))

    def paintEvent(self, _event: QPaintEvent) -> None:

        painter = QPainter(self)

        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            if self.draw_mode == DrawMode.BOX:
                for d in self.detections:
                    painter.drawRect(*d["box"])
                    return

            h, w = self.frame.shape[:2]

            if self.draw_mode == DrawMode.BLUR:
                image = cv2.GaussianBlur(
                    src=self.frame,
                    ksize=(51, 51),
                    sigmaX=0,
                )

            elif self.draw_mode == DrawMode.PIXELATE:
                small = cv2.resize(
                    src=self.frame,
                    dsize=(
                        max(1, w // self.pixelate_block_size),
                        max(1, h // self.pixelate_block_size),
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )

                image = cv2.resize(
                    src=small,
                    dsize=(w, h),
                    interpolation=cv2.INTER_NEAREST,
                )

            qimage = QImage(
                image.data,
                w,
                h,
                image.strides[0],
                QImage.Format.Format_RGB888,
            )

            painter.setClipRegion(self.draw_region)
            painter.drawImage(0, 0, qimage)

        finally:
            painter.end()
