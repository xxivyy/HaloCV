import logging
import struct
import time

import numpy as np
from fastapi import WebSocket

from halocv import HaloCV

model = HaloCV()

logger = logging.getLogger("protocol")


class Protocol:
    def __init__(self, websocket: WebSocket, version: str) -> None:
        self.websocket = websocket
        self.version = version

        self.fps = 0
        self.fps_last = time.perf_counter()

    async def handshake(self) -> None:
        await self.websocket.send_json(
            {
                "version": self.version,
                "resolution": model.resolution,
                "compress": False,  # TODO.
            }
        )

    async def process(self) -> None:

        ## -- 1: Parse Payload -- ##

        payload = await self.websocket.receive_bytes()

        width, height = struct.unpack_from("!HH", payload, 0)

        frame = np.frombuffer(
            payload,
            dtype=np.uint8,
            offset=4,
        ).reshape(height, width, 3)

        ## -- 2: Model Prediction -- ##

        prediction = model.predict(
            frame,
            conf=0.05,
            iou=0.45,
        )

        await self.websocket.send_json(prediction)

        ## -- 3: FPS Calculation -- ##

        self.fps += 1
        now = time.perf_counter()

        if now - self.fps_last >= 1.0:
            logger.info(f"FPS: {self.fps}")
            self.fps = 0
            self.fps_last = now
