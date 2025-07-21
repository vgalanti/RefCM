import os
import logging

from .config import LOG_FILE

log = logging.getLogger(__name__)


def start_logging(console_level: int = logging.INFO) -> None:
    """
    Starts logger & sets the console and file handlers and thresholds.

    Parameters
    ----------
    console_level: int = logging.INFO
        Logging level threshold for console display.
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # logging DEBUG and higher to file
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(name)-16s] [%(levelname)-8s] : %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
        filename=LOG_FILE,
        filemode="w",
    )

    # console_level messages and higher to sys.stderr
    console = logging.StreamHandler()
    console.setLevel(console_level)
    formatter = logging.Formatter("[%(name)-16s] [%(levelname)-8s] : %(message)s")
    console.setFormatter(formatter)

    # add console handler to the root logger
    logging.getLogger("").addHandler(console)

    logging.getLogger("numba").setLevel(logging.ERROR)
