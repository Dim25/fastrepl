import os
from inspect import FrameInfo


class LocalContext:
    __slots__ = ("_filename", "_function")

    _filename: str
    _function: str

    def __init__(self, frame: FrameInfo) -> None:
        self._filename, self._function = (
            # TODO: This can be problem if two files in different directory has same filename and Updatable.key
            os.path.basename(frame.filename),
            frame.function,
        )

    def __str__(self) -> str:
        return f"{self._filename}:{self._function}"

    def __repr__(self) -> str:
        return f"VariableContext({self._filename!r}, {self._function!r})"

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
