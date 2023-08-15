from contextlib import ContextDecorator
from graphviz import Digraph

from fastrepl.utils import (
    GraphInfo,
    build_graph as _build_graph,
    get_cuid,
    pairwise,
    OrderedSet,
)
from fastrepl.context import GlobalContext, LocalContext


class AnalyzeController:
    @staticmethod
    def next_run():
        GlobalContext.next_run()

    @staticmethod
    def build_report():
        raise NotImplementedError

    @staticmethod
    def convert_graph(GRAPH=2) -> GraphInfo:
        if GRAPH < 1:
            raise ValueError("GRAPH is not enabled.")

        info = GraphInfo(id=get_cuid(), nodes=[], edges=[])

        def _get_node_name(ctx: LocalContext):
            # NOTE: : is reserved in graphviz
            return str(ctx).replace(":", "_")

        def _get_node_label(ctx: LocalContext, keys: OrderedSet[str]):
            if GRAPH >= 3:
                return f"{str(ctx)}\n{keys}"
            elif GRAPH >= 2:
                return f"{ctx.function}\n{keys}"
            elif GRAPH >= 1:
                return ctx.function

        for run in GlobalContext.run_ctx_keys:
            for ctx, keys in run.items():
                info["nodes"].append((_get_node_name(ctx), _get_node_label(ctx, keys)))
            for ctx_a, ctx_b in pairwise(run.keys()):
                info["edges"].append((_get_node_name(ctx_a), _get_node_name(ctx_b)))

        if GRAPH >= 3 or len(GlobalContext.run_ctx_keys) > 1:
            for i, run in enumerate(GlobalContext.run_ctx_keys):
                first_ctx = list(run.keys())[0]
                info["nodes"].append((f"run_{i}", f"run_{i}"))
                info["edges"].append((f"run_{i}", _get_node_name(first_ctx)))
        return info

    @staticmethod
    def build_graph(GRAPH=2) -> Digraph:  # pragma: no cover
        return _build_graph(AnalyzeController.convert_graph(GRAPH=GRAPH))


class Analyze(ContextDecorator):
    def __init__(self):
        self.controller = AnalyzeController()

    def __enter__(self) -> AnalyzeController:
        return self.controller

    def __exit__(self, *args):
        GlobalContext.reset_analyze()
