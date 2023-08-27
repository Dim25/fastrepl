from typing import TypedDict, Tuple, List

from graphviz import Digraph


class GraphInfo(TypedDict):
    id: str
    nodes: List[Tuple[str, str]]
    edges: List[Tuple[str, str]]


def build_graph(graph: GraphInfo) -> Digraph:  # pragma: no cover
    PLAIN_PREFIX = "run_"

    dot = Digraph(name=graph["id"], comment=graph["id"], format="png")
    dot.attr(label=f'status: {graph["id"]}')

    for name, label in graph["nodes"]:
        shape = "plain" if str.startswith(name, PLAIN_PREFIX) else "ellipse"
        dot.node(name, label, shape=shape)
    for a, b in graph["edges"]:
        dot.edge(a, b)
    return dot
