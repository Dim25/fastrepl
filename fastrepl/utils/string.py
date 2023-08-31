def truncate(text: str, max: int) -> str:
    if len(text) <= max:
        return text

    if max <= 3:
        raise ValueError("max must be at least 4")

    return text[: max - 3] + "..."
