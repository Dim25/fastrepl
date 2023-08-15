from typing import Generic, TypeVar, Dict, OrderedDict


OrderedSetItem = TypeVar("OrderedSetItem")


class OrderedSet(Generic[OrderedSetItem]):
    def __init__(self):
        self.data = OrderedDict[OrderedSetItem, None]()

    def add(self, item: OrderedSetItem):
        self.data[item] = None

    def keys(self):
        return list(self.data.keys())

    def __str__(self) -> str:  # pragma: no cover
        return "[" + ", ".join(self.data.keys()) + "]"

    def __repr__(self) -> str:  # pragma: no cover
        return f"OrderedSet: {self.__str__()}"

    def __eq__(self, _) -> bool:
        raise NotImplementedError

    def __contains__(self, item):
        return item in self.data

    def __len__(self):
        return len(self.data)


HistoryDictItem = TypeVar("HistoryDictItem")


class HistoryDict(OrderedDict[str, HistoryDictItem]):
    def __init__(self, initial: HistoryDictItem):
        raise NotImplementedError
