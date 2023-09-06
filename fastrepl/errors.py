from abc import ABC, abstractmethod


class Error(Exception, ABC):
    def __init__(self, msg="") -> None:
        self.msg = msg

    def __str__(self) -> str:
        if self.msg == "":
            return self.doc_url()
        return f"{self.msg} | {self.doc_url()}"

    @abstractmethod
    def doc_url(self) -> str:
        ...


class InvalidStatusError(Error):
    def doc_url(self) -> str:
        return "https://docs.fastrepl.com"  # pragma: no cover


class EmptyGraphError(Error):
    def doc_url(self) -> str:
        return "https://docs.fastrepl.com"  # pragma: no cover


class EmptyPipelineError(Error):
    def doc_url(self) -> str:
        return "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#emptypipeline"  # pragma: no cover


class EmptyPredictionsError(Error):
    def doc_url(self) -> str:
        return "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#emptypredictions"  # pragma: no cover


class NoneReferenceError(Error):
    def doc_url(self) -> str:
        return "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#nonereference"  # pragma: no cover


class TokenizeNotImplementedError(Error, NotImplementedError):
    def doc_url(self) -> str:
        return "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#tokenizenotimplemented"  # pragma: no cover
