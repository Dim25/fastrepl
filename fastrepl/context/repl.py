import warnings
from typing import ClassVar, OrderedDict

from fastrepl.context.local import LocalContext
from fastrepl.utils import get_cuid


class REPLContext:
    latest_id: ClassVar[str] = get_cuid()
    head_id: ClassVar[str] = latest_id
    ctx_key_values: ClassVar[dict[LocalContext, dict[str, OrderedDict[str, str]]]] = {}

    @staticmethod
    def reset():
        REPLContext.ctx_key_values = {}

    @staticmethod
    def trace(ctx: LocalContext, key: str, value: str):
        if ctx not in REPLContext.ctx_key_values:
            REPLContext.ctx_key_values[ctx] = {}
        if key not in REPLContext.ctx_key_values[ctx]:
            REPLContext.ctx_key_values[ctx][key] = OrderedDict()
        REPLContext.ctx_key_values[ctx][key][REPLContext.latest_id] = value

    # TODO: This unlikely the final API, we might use Runner to update the value.
    @staticmethod
    def update(target_ctx: LocalContext, target_key: str, new_value: str):
        for ctx, key_values in REPLContext.ctx_key_values.items():
            for key, values in key_values.items():
                if ctx == target_ctx and key == target_key:
                    values[REPLContext.latest_id] = new_value
                else:
                    # TODO: Here we just copy the previous value
                    previous = next(reversed(values.values()))
                    values[REPLContext.latest_id] = previous

    @staticmethod
    def get_current_value(ctx: LocalContext, key: str) -> str:
        history = REPLContext.ctx_key_values[ctx][key]
        ret = history.get(REPLContext.head_id, None)
        if ret is None:
            k, v = next(reversed(history.items()))
            warnings.warn(
                f"when retrieving {key!r}, {REPLContext.head_id!r} not found. Using {k!r} instead",
                UserWarning,
            )
            return v
        return ret
