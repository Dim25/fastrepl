from typing import List, Any, Literal, get_args

import evaluate

from fastrepl.warnings import warn, IncompletePredictionWarning
from fastrepl.errors import NoneReferenceError
from fastrepl.eval.base import BaseEvalWithReference

HUGGINGFACE_BUILTIN_METRICS = Literal[
    "precision",
    "code_eval",
    "roc_auc",
    "cuad",
    "xnli",
    "rouge",
    "pearsonr",
    "mse",
    "super_glue",
    "comet",
    "cer",
    "sacrebleu",
    "mahalanobis",
    "wer",
    "competition_math",
    "f1",
    "recall",
    "coval",
    "mauve",
    "xtreme_s",
    "bleurt",
    "ter",
    "accuracy",
    "exact_match",
    "indic_glue",
    "spearmanr",
    "mae",
    "squad",
    "chrf",
    "glue",
    "perplexity",
    "mean_iou",
    "squad_v2",
    "meteor",
    "bleu",
    "wiki_split",
    "sari",
    "frugalscore",
    "google_bleu",
    "bertscore",
    "matthews_correlation",
    "seqeval",
    "trec_eval",
    "rl_reliability",
    "poseval",
    "brier_score",
    "mase",
    "mape",
    "smape",
    "nist_mt",
    "character",
    "charcut_mt",
    "r_squared",
]

HUGGINGFACE_FASTREPL_METRICS = Literal["mean_reciprocal_rank", "mean_average_precision"]


class HuggingfaceMetric(BaseEvalWithReference):
    __slots__ = ("name", "module")

    def __init__(self, name: str) -> None:
        self.name = name

        CURRENTLY_SUPPORTED = [
            "exact_match",
            "f1",
            "recall",
            "precision",
            "accuracy",
            "matthews_correlation",
            "mse",
            "mae",
            "rouge",
            "bleu",
        ]

        if name in get_args(HUGGINGFACE_FASTREPL_METRICS):
            raise NotImplementedError(
                f"we have it here: 'https://huggingface.co/spaces/fastrepl/{name}', but not implemented yet."
            )
            # self.module = evaluate.load(f"fastrepl/{name}")
        else:
            if name not in CURRENTLY_SUPPORTED:
                raise NotImplementedError(
                    f"Huggingface has it here: 'https://huggingface.co/spaces/evaluate-metric/{name}', but we don't support it at the moment."
                )
            self.module = evaluate.load(name)

    def _validate(self, predictions: List[Any], references: List[Any]) -> None:
        if any(v is None for v in predictions):
            warn(IncompletePredictionWarning)
        if any(v is None for v in references):
            raise NoneReferenceError

    def compute(self, predictions: List[Any], references: List[Any], **kwargs):
        self._validate(predictions, references)

        result = self.module.compute(
            predictions=predictions, references=references, **kwargs
        )
        # Huggingface has some inconsistencies in their API. Fix here if needed.
        return result
