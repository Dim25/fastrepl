from rich.pretty import pprint

from fastrepl.utils import Variable, console  # type: ignore[no-redef]


DEBUG = Variable("DEBUG", 0)


def debug(input) -> None:  # pragma: no cover
    if DEBUG < 1:
        return None

    if DEBUG > 1:
        pprint(input, expand_all=True)
    else:
        pprint(input, expand_all=True, max_string=110)
