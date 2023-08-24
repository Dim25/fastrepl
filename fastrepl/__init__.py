from typing import Optional

from fastrepl.context import REPLContext, AnalyzeContext, LocalContext
from fastrepl.analyze import Analyze
from fastrepl.repl import REPL, REPLController
from fastrepl.run.cache import BaseCache

load_report = REPLController.load_report
cache: Optional[BaseCache] = None

__all__ = [
    "REPLContext",
    "AnalyzeContext",
    "LocalContext",
    "Analyze",
    "REPL",
    "load_report",
]
