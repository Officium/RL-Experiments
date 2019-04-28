"""
Logger singleton wrapper
Default logger folder is `os.path.join(__file__, '..', '..', 'logs')`
"""
import logging
import logging.handlers
import os


__all__ = ['get_logger']


class Singleton(type):
    _instances = dict()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = \
                super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class _Logger(object):
    __metaclass__ = Singleton

    logDir = os.path.join(os.path.curdir, '..', 'logs')
    defaultKey = 'log'
    os.makedirs(logDir, exist_ok=True)
    logLevel = logging.INFO
    logFormat = '%(levelname)s %(asctime)s %(processName)s %(process)s' \
                '| %(filename)s | [line:%(lineno)d] | %(message)s'
    logMaxByte = 0  # 20 * 1024 * 1024
    logBackupCount = 0  # 10

    def __init__(self):
        self._loggers = dict()
        os.makedirs(_Logger.logDir, exist_ok=True)
        self._add_logger(_Logger.defaultKey)

    def _add_logger(self, key):
        # Reference: https://docs.python.org/2/library/logging.handlers.html
        logger = logging.getLogger(key)
        logger.setLevel(_Logger.logLevel)
        path = os.path.join(_Logger.logDir, key)

        # file handler (log file)
        log_handler = logging.handlers.RotatingFileHandler(
            filename=path,
            maxBytes=_Logger.logMaxByte,
            backupCount=_Logger.logBackupCount
        )
        log_handler.setLevel(_Logger.logLevel)
        log_handler.setFormatter(logging.Formatter(_Logger.logFormat))
        logger.addHandler(log_handler)

        # # stream handler (default sys.stderr)
        # log_handler = logging.StreamHandler()
        # log_handler.setLevel(_Logger.logLevel)
        # log_handler.setFormatter(logging.Formatter(_Logger.logFormat))
        # logger.addHandler(log_handler)
        self._loggers[key] = logger

    def __getitem__(self, key):
        if key is None:
            key = _Logger.defaultKey
        else:
            key = str(key)
            if self._loggers.get(key) is None:
                self._add_logger(key)
        return self._loggers[key]


def get_logger(key=None):
    """ Get logger by key, key can be different random seed or algorithms """
    return _Logger()[key]
