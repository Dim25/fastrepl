from typing import Optional, TextIO, Dict
import functools

from rich.prompt import Prompt
from rich.console import Console

from fastrepl.eval.base import BaseEvalWithoutReference


class HumanClassifierRich(BaseEvalWithoutReference):
    def __init__(
        self,
        labels: Dict[str, str],
        instruction: str = "Classify the following sample",
        prompt_template="[bright_magenta]{instruction}:[/bright_magenta]\n\n{sample}\n\n",
        console: Optional[Console] = None,
        stream: Optional[TextIO] = None,
    ) -> None:
        self.labels = labels
        self.render_prompt = functools.partial(
            prompt_template.format, instruction=instruction
        )
        self.console = console
        self.stream = stream

    def compute(self, sample: str, context=None) -> str:
        prompt = self.render_prompt(sample=sample)
        choices = list(self.labels.keys())  # TODO: Render descriptions

        if context is None:
            return Prompt.ask(
                prompt,
                choices=choices,
                console=self.console,
                stream=self.stream,
            )
        else:
            return Prompt.ask(
                prompt,
                choices=choices,
                default=context,
                console=self.console,
                stream=self.stream,
            )

    def is_interactive(self) -> bool:
        return True
