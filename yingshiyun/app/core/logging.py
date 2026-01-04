import logging


def configure_logging(level: str = "INFO"):
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
