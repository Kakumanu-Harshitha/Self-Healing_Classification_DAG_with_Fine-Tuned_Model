# logging_config.py
import logging
from pythonjsonlogger import jsonlogger

def get_json_logger(name="self_heal"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("run_log.json")
    fmt = '%(asctime)s %(levelname)s %(name)s %(message)s'
    json_formatter = jsonlogger.JsonFormatter(fmt)
    handler.setFormatter(json_formatter)
    logger.addHandler(handler)
    # also console
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)
    return logger
