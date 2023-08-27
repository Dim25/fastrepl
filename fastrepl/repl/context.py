from typing import (
    ClassVar,
    Tuple,
    List,
    Dict,
    DefaultDict,
)

from graphviz import Digraph

from fastrepl.errors import (
    InvalidStatusError,
    EmptyGraphError,
)
from fastrepl.utils import (
    LocalContext,
    GraphInfo,
    build_graph,
    get_cuid,
    pairwise,
)


def set_status(status: str):
    REPLContext.set_status(status)


def update(pairs: List[Tuple[str, str]]):
    REPLContext.update(pairs)


def graph(level=2) -> Digraph:
    nodes: List[Tuple[str, str]] = []
    edges: List[Tuple[str, str]] = []

    def _get_node_name(ctx: LocalContext):
        # NOTE: : is reserved in graphviz
        return str(ctx).replace(":", "_")

    def _get_node_label(ctx: LocalContext, keys: List[str]):
        if level >= 3:
            return f"{str(ctx)}\n{keys}"
        elif level >= 2:
            return f"{ctx.function}\n{keys}"
        elif level >= 1:
            return ctx.function

    for ctx, key_values in REPLContext._trace.items():
        name, label = _get_node_name(ctx), _get_node_label(ctx, list(key_values.keys()))
        nodes.append((name, label))
    for ctx_a, ctx_b in pairwise(REPLContext._trace.keys()):
        edges.append((_get_node_name(ctx_a), _get_node_name(ctx_b)))

    if len(nodes) < 1:
        raise EmptyGraphError()

    return build_graph(GraphInfo(id=REPLContext._status, nodes=nodes, edges=edges))


class REPLContext:
    """single context that tracks all experiments in a REPL"""

    _status: ClassVar[str] = get_cuid()
    """current status of the REPL"""
    _history: ClassVar[List[str]] = [_status]
    """history of REPL status"""
    # fmt: off
    _trace: Dict[LocalContext, Dict[str, Dict[str, str]]] = DefaultDict(lambda: DefaultDict(dict))
    """mapping: ctx -> key -> status -> value"""
    # fmt: on

    @staticmethod
    def reset():
        REPLContext._status = get_cuid()
        REPLContext._history = [REPLContext._status]
        REPLContext._trace = DefaultDict(lambda: DefaultDict(dict))

    @staticmethod
    def trace(ctx: LocalContext, key: str, value: str):
        """
        try-except will prevent re-initialization the value of key after update
        TODO: We can not prevent user to provide duplicated key
        """
        try:
            _ = REPLContext._trace[ctx][key][REPLContext._status]
        except KeyError:
            REPLContext._trace[ctx][key][REPLContext._status] = value

    @staticmethod
    def update(pairs: List[Tuple[str, str]]):
        # TODO: Warn user if pairs contain invalid key

        next_status = get_cuid()

        for key_status_value in REPLContext._trace.values():
            for key, status_value in key_status_value.items():
                try:
                    _, new_value = next((k, v) for k, v in pairs if key == k)
                    status_value[next_status] = new_value
                except StopIteration:
                    status_value[next_status] = status_value[REPLContext._status]

        REPLContext._history.append(next_status)
        REPLContext._status = next_status

    @staticmethod
    def set_status(status: str):
        if status not in REPLContext._history:
            raise InvalidStatusError()

        REPLContext._status = status

    @staticmethod
    def get_current(ctx: LocalContext, key: str) -> str:
        return REPLContext._trace[ctx][key][REPLContext._status]
