from typing import Optional, Union, Any


def truncate(text: str, max: int) -> str:
    if len(text) <= max:
        return text

    if max <= 3:
        raise ValueError("max must be at least 4")

    return text[: max - 3] + "..."


def number(s: Any) -> Optional[Union[int, float]]:
    if isinstance(s, int) or isinstance(s, float):
        return s

    if not isinstance(s, str):
        return None

    s = s.strip()

    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None
