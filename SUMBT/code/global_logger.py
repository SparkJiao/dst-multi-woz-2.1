import logging
from tensorboardX import SummaryWriter

logger = logging.getLogger()
summary_writer = None


def register_logger(log: logging.Logger):
    global logger
    logger = log


def get_child_logger(name):
    return logger.getChild(name)


def register_summary_writer(writer: SummaryWriter):
    global summary_writer
    summary_writer = writer


def get_summary_writer():
    return summary_writer
