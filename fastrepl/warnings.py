import warnings
from typing import cast


def warning_formatter(message, category, filename, lineno, line=None):
    if not hasattr(category, "fastrepl"):  # NOTE: this is built-in warning formatting
        msg = warnings.WarningMessage(message, category, filename, lineno, None, line)
        return warnings._formatwarnmsg_impl(msg)

    category = cast(Warning, category)

    if str(message) == "":
        return f"{filename}:{lineno}: {category.__name__} | {category.doc_url()}\n"

    return (
        f"{filename}:{lineno}: {category.__name__}: {message} | {category.doc_url()}\n"
    )


warnings.formatwarning = warning_formatter


def warn(category=Warning, context=""):
    warnings.warn(context, category)


class Warning(UserWarning):
    @staticmethod
    def fastrepl() -> bool:
        return True

    @staticmethod
    def doc_url() -> str:
        raise NotImplementedError


class VerbosityBiasWarning(Warning):
    @staticmethod
    def doc_url() -> str:
        return (
            "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#verbositybias"
        )


class InvalidPredictionWarning(Warning):
    @staticmethod
    def doc_url() -> str:
        return "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#invalidprediction"


class IncompletePredictionWarning(Warning):
    @staticmethod
    def doc_url() -> str:
        return "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#incompleteprediction"


class CompletionTruncatedWarning(Warning):
    @staticmethod
    def doc_url() -> str:
        return "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#completiontruncated"


class UnknownLLMExceptionWarning(Warning):
    @staticmethod
    def doc_url() -> str:
        return "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#unknownllmexception"


class FloatGradingWarning(Warning):
    @staticmethod
    def doc_url() -> str:
        return (
            "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#floatgrading"
        )
