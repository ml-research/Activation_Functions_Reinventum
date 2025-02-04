import logging
import os
import copy


# https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/
class ColoredFormatter(logging.Formatter):
    reset = "\x1b[0m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    white = "\x1b[38;5;231m"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.colors = {
            logging.DEBUG: self.blue,
            logging.INFO: self.white,
            logging.CRITICAL: self.bold_red,
            logging.ERROR: self.red,
            logging.WARNING: self.yellow,
        }

    def format(self, record):
        record = copy.copy(record)
        record.msg = self.colors[record.levelno] + record.msg + self.reset
        return super().format(record)


class ActivationLogger(object):
    def __init__(
        self,
        logger_name,
        log_level=logging.DEBUG,
        show_logger_name=True,
        show_time=False,
    ):
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(log_level)
        console = logging.StreamHandler()
        console.setLevel(log_level)

        message_format = self.get_format(show_logger_name, show_time)
        formatter = logging.Formatter(message_format)

        if os.name != "nt":
            console.setFormatter(ColoredFormatter(message_format))
        if os.name == "nt":
            console.setFormatter(formatter)

        self._logger.addHandler(console)

    def debug(self, msg):
        self._logger.debug(msg)

    def warn(self, msg):
        self._logger.warn(msg)

    def info(self, msg):
        self._logger.info(msg)

    def error(self, msg):
        self._logger.error(msg)

    def critical(self, msg):
        self._logger.critical(msg)

    def get_format(self, show_logger_name, show_time):
        format = ""

        if show_logger_name:
            format = "%(name)s"

        if show_time:
            if len(format) > 0:
                format = format + " | "

            format = format + "%(asctime)s"

        if len(format) > 0:
            format = format + " | "

        format = format + "%(filename)s | %(lineno)d | %(message)s"

        return format
