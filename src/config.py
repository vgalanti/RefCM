import gdown
import logging

DRIVE_ID = "1fWWaxBLUdacBT9r-1CymdyRICMPStvBJ"
LOG_FILE = "log.log"
DB_FPATH = "db.json"
DATA_PATH = "data"
MATCHING_FILE = "matchings.csv"
MATCHING_COSTS_FILE = "matching_costs.csv"

lg = logging.getLogger(__name__)


def start_logging(console_level: int = logging.INFO) -> None:
    """
    Starts logger & sets the console and file handlers and thresholds.

    Parameters
    ----------
    console_level: int = logging.INFO
        Logging level threshold for console display.
    """
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


def load_data_from_drive() -> None:
    """
    Downloads datasets from google drive folder listed in config file.
    """
    lg.info(f"starting download from gdrive")
    gdown.download_folder(id=DRIVE_ID)
    lg.info("drive download complete")
