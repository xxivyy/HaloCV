from fastapi import WebSocket


class Protocol:
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket

    async def handshake(self) -> None:
        await self.websocket.accept()

        # TODO: validate version & config.

    async def process(self) -> None:
        await self.websocket.receive_bytes()

        # TODO: process received frame.
