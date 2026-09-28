from contextlib import asynccontextmanager

from fastapi import FastAPI


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
