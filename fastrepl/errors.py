class Error(Exception):
    pass


class InvalidStatusError(Error):
    pass


class EmptyGraphError(Error):
    pass


class NonePredictionError(Error):
    pass


class NoneReferenceError(Error):
    pass
