import uvicorn

from server import app


def main() -> None:

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        # log_config=None,
        # access_log=False,
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
