from typing import Callable, Any
import functools

# fmt: off
def ensure(check: Callable[[Any], bool]):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            assert check(*args, **kwargs)
            return fn(*args, **kwargs)
        return wrapper
    return decorator
# fmt: on
