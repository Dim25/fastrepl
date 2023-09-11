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
    try:  # TODO: to avoid `cannot use a string pattern on a bytes-like object`. this sometimes raises, not sure why
        warnings.warn(context, category)
    except:
        warnings.warn("", category)


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
        return "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#verbositybias"  # pragma: no cover


class InvalidPredictionWarning(Warning):
    @staticmethod
    def doc_url() -> str:
        return "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#invalidprediction"  # pragma: no cover


class InconsistentPredictionWarning(Warning):
    @staticmethod
    def doc_url() -> str:
        return "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#inconsistentprediction"  # pragma: no cover


class IncompletePredictionWarning(Warning):
    @staticmethod
    def doc_url() -> str:
        return "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#incompleteprediction"  # pragma: no cover


class CompletionTruncatedWarning(Warning):
    @staticmethod
    def doc_url() -> str:
        return "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#completiontruncated"  # pragma: no cover


class UnknownLLMExceptionWarning(Warning):
    @staticmethod
    def doc_url() -> str:
        return "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#unknownllmexception"  # pragma: no cover


class FloatGradingWarning(Warning):
    @staticmethod
    def doc_url() -> str:
        return "https://docs.fastrepl.com/miscellaneous/warnings_and_errors#floatgrading"  # pragma: no cover
