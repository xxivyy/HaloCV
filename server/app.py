import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect

from .protocol import Protocol


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Server",
    version="0.1.0",
    openapi_url=None,
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

router = APIRouter(include_in_schema=False)
logger = logging.getLogger("ws")


@router.websocket("/")
async def ws_root(websocket: WebSocket) -> None:
    await websocket.accept()

    connection = Protocol(websocket, app.version)
    await connection.handshake()

    logger.info("client connected")

    try:
        while True:
            await connection.process()

    except WebSocketDisconnect:
        logger.info("client disconnected")


app.include_router(router)
