import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .protocol import Protocol

router = APIRouter(include_in_schema=False)
logger = logging.getLogger("ws")


@router.websocket("/")
async def ws_root(websocket: WebSocket) -> None:

    connection = Protocol(websocket)
    await connection.handshake()

    logger.info("client connected")

    try:
        while True:
            await connection.process()

    except WebSocketDisconnect:
        logger.info("client disconnected")
