import os
import functools
import dotenv


def loadenv():  # pragma: no cover
    dotenv.load_dotenv()


def setenv(key, value):  # pragma: no cover
    os.environ[key] = str(value)


@functools.lru_cache(maxsize=None)
def getenv(key, default=0):  # pragma: no cover
    return type(default)(os.getenv(key, default))
