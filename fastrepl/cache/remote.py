from fastrepl.cache.base import BaseCache


class RemoteCache(BaseCache):
    def __init__(self) -> None:
        raise NotImplementedError
