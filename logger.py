import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """
    Set up a single stdout handler on the root logger.
    Calling this multiple times is safe — it bails out if handlers already exist.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # already configured, don't add duplicate handlers

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(handler)
