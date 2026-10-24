import time
from collections import deque
from threading import Event, Thread

import cv2
import dxcam
import numpy as np
from PySide6.QtCore import QObject, Signal

from halocv import HaloCV

from .overlay import Overlay


class Detector(QObject):
    fps_signal = Signal(int)

    overlay = Overlay()

    camera = dxcam.create(
        output_color="RGB",
        backend="dxgi",
        processor_backend="cv2",
    )

    def __init__(
        self,
        model_path: str,
        conf: float,
        iou: float,
    ) -> None:
        super().__init__()

        self.model = HaloCV(model_path)
        self.stop_event = Event()

        self.conf = conf
        self.iou = iou

        self.fps_times: deque[float] = deque()
        self.fps_last = time.perf_counter()

        thread = Thread(target=self.loop, daemon=True)
        thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def loop(self) -> None:

        ## -- 1: Setup Resolution -- ##

        while True:
            frame = self.camera.grab(copy=False, new_frame_only=True)

            if frame is not None:
                screen_h, screen_w = frame.shape[:2]
                break

        scale = min(
            1.0,
            self.model.input_height / screen_h,
            self.model.input_width / screen_w,
        )

        input_h = max(1, round(screen_h * scale))
        input_w = max(1, round(screen_w * scale))

        buffer = np.empty(
            (input_h, input_w, 3),
            dtype=np.uint8,
        )

        ## -- 2: Enter Detection Loop -- ##

        while not self.stop_event.is_set():
            frame = self.camera.grab(copy=False, new_frame_only=True)

            if frame is None:
                continue

            if frame.shape[:2] != (screen_h, screen_w):
                raise RuntimeError("Monitor resolution changed.")

            cv2.resize(
                src=frame,
                dsize=(input_w, input_h),
                dst=buffer,
                interpolation=cv2.INTER_AREA,
            )

            detections = self.model.detect(
                frame,
                conf=self.conf,
                iou=self.iou,
                box_resolution=(screen_w, screen_h),
            )

            self.overlay.draw_signal.emit(detections, frame)

            fps_now = time.perf_counter()
            self.fps_times.append(fps_now)

            while fps_now - self.fps_times[0] > 1.0:
                self.fps_times.popleft()

            if fps_now - self.fps_last >= 1:
                self.fps_signal.emit(len(self.fps_times))
                self.fps_last = fps_now
