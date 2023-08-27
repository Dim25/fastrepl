from inspect import FrameInfo


class LocalContext:
    __slots__ = ("_filename", "_function")

    def __init__(self, frame: FrameInfo) -> None:
        self._filename = frame.filename
        self._function = frame.function

    def __str__(self) -> str:  # pragma: no cover
        return f"{self._filename}:{self._function}"

    def __repr__(self) -> str:  # pragma: no cover
        return f"LocalContext({self._filename!r}, {self._function!r})"

    def __hash__(self) -> int:
        return hash((self._filename, self._function))

    def __eq__(self, v: object) -> bool:
        if not isinstance(v, LocalContext):
            return False
        return self._filename == v._filename and self._function == v._function

    @property
    def filename(self):
        return self._filename

    @property
    def function(self):
        return self._function
