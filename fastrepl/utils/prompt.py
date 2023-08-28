# stripped down from https://github.com/normal-computing/outlines/blob/ee941fd60749d88ea92746935462edc253d53cd6/outlines/text/prompts.py

import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, cast

from jinja2 import Environment, StrictUndefined


@dataclass
class Prompt:
    template: str
    signature: inspect.Signature

    def __post_init__(self) -> None:
        self.parameters: List[str] = list(self.signature.parameters.keys())

    def __call__(self, *args, **kwargs) -> str:
        bound_arguments = self.signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        return render(self.template, **bound_arguments.arguments)

    def __str__(self):
        return self.template


def prompt(fn: Callable) -> Prompt:
    signature = inspect.signature(fn)

    docstring = fn.__doc__
    if docstring is None:
        raise TypeError("Could not find a template in the function's docstring.")

    template = cast(str, docstring)

    return Prompt(template, signature)


def render(template: str, **values: Optional[Dict[str, Any]]) -> str:
    # Dedent, and remove extra linebreak
    cleaned_template = inspect.cleandoc(template)

    # Add linebreak if there were any extra linebreaks that
    # `cleandoc` would have removed
    ends_with_linebreak = template.replace(" ", "").endswith("\n\n")
    if ends_with_linebreak:
        cleaned_template += "\n"

    # Remove extra whitespaces, except those that immediately follow a newline symbol.
    # This is necessary to avoid introducing whitespaces after backslash `\` characters
    # used to continue to the next line without linebreak.
    cleaned_template = re.sub(r"(?![\r\n])(\b\s+)", " ", cleaned_template)

    env = Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )

    jinja_template = env.from_string(cleaned_template)

    return jinja_template.render(**values)
