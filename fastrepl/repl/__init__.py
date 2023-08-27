from fastrepl.repl.context import graph, set_status, update
from fastrepl.repl.polish import Updatable

import fastrepl.llm as llm
import fastrepl.eval as eval
import fastrepl.cache as cache
from fastrepl.cache import llm_cache

from fastrepl.runner import (
    LocalRunnerREPL as LocalRunner,
    RemoteRunnerREPL as RemoteRunner,
)

from fastrepl.errors import (
    InvalidStatusError,
    EmptyGraphError,
)

from contextvars import ContextVar
from contextlib import contextmanager


IS_REPL = ContextVar[bool]("IS_REPL", default=False)


@contextmanager
def REPL():
    IS_REPL.set(True)
    try:
        yield
    finally:
        IS_REPL.set(False)
