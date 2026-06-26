import uvicorn


def main() -> None:
    uvicorn.run(
        "vram_fabric.api.routes:app",
        host="0.0.0.0",
        port=8081,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
