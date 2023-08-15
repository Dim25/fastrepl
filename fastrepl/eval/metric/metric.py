from typing import Any
import evaluate


class BaseMetric:
    def __init__(self, name: str) -> None:
        self.name = name

    def compute(self, predictions, references, **kwargs):
        raise NotImplementedError


class HuggingfaceMetric(BaseMetric):
    def __init__(self, name: str) -> None:
        # TODO: https://github.com/huggingface/evaluate/tree/main/metrics
        NOT_SUPPORTED = [
            # Need additional package
            "frugalscore",
            "competition_math",
            "cer",
            "wer",
            "mauve",
            "ter",
            "coval",
            "rouge",
            "sacrebleu",
            "comet",
            "bleurt",
            "meteor",
            "chrf",
            "bertscore",
            "wiki_split",
            "sari",
            "seqeval",
            "charcut_mt",
            "trec_eval",
            "rl_reliability",
            "character",
            # Need configuration
            "indic_glue",
            "xtreme_s",
            "super_glue",
            "glue",
            # Others
            "mahalanobis",
            "roc_auc",
            "code_eval",
            "cuad",
            "perplexity",
            "mean_iou",
            "squad",
            "squad_v2",
            "poseval",
            "mase",
            "bleu",
            "google_bleu",
            "exact_match",
            "nist_mt",
        ]

        # TODO
        CLASSIFICATION_METRICS = [
            "precision",
            "xnli",
            "pearsonr",
            "mse",
            "f1",
            "recall",
            "accuracy",
            "spearmanr",
            "mae",
            "matthews_correlation",
            "brier_score",
            "mape",
            "smape",
            "r_squared",
        ]

        if name in NOT_SUPPORTED or name in CLASSIFICATION_METRICS:
            raise NotImplementedError

        super().__init__(name)
        self.metric = evaluate.load(name)

    def compute(self, predictions, references, **kwargs) -> Any:  # TODO
        return self.metric.compute(
            predictions=predictions, references=references, **kwargs
        )
