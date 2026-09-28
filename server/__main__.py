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
        ws_max_size=16 * 1024 * 1024,
        ws_max_queue=16,
        ws_ping_interval=20,
        ws_ping_timeout=10,
        ws_per_message_deflate=False,
    )


if __name__ == "__main__":
    main()
