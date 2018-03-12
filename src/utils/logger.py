# -*- coding: utf-8 -*-
import logging
import logging.handlers
import os


def get_logger():
    return _Logger().logger


class Singleton(type):
    _instances = dict()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class _Logger(object):
    __metaclass__ = Singleton

    logLevel = logging.INFO
    logFile = __file__ + os.sep.join(('', '..', '..', 'logs', 'log'))
    logFormatterStr = '%(levelname)s %(asctime)s module %(process)s| %(processName)s| ' \
                      '%(filename)s| [line:%(lineno)d]| %(message)s'
    logMaxByte = 20 * 1024 * 1024
    logBackupCount = 10

    def __init__(self):
        # Reference: https://docs.python.org/2/library/logging.handlers.html
        self.logger = logging.getLogger()
        self.logger.setLevel(_Logger.logLevel)

        # file handler (log file)
        log_handler = logging.handlers.RotatingFileHandler(filename=_Logger.logFile,
                                                           maxBytes=_Logger.logMaxByte,
                                                           backupCount=_Logger.logBackupCount)
        log_handler.setLevel(_Logger.logLevel)
        log_handler.setFormatter(logging.Formatter(_Logger.logFormatterStr))
        self.logger.addHandler(log_handler)

        # stream handler (default sys.stderr)
        log_handler = logging.StreamHandler()
        log_handler.setLevel(_Logger.logLevel)
        log_handler.setFormatter(logging.Formatter(_Logger.logFormatterStr))
        self.logger.addHandler(log_handler)
