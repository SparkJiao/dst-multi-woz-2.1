import logging

logger = logging.getLogger()


def register_logger(log: logging.Logger):
    global logger
    logger = log


def get_child_logger(name):
    return logger.getChild(name)
