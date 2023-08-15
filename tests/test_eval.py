import pytest
from evaluate import list_evaluation_modules

from fastrepl.eval.metric.metric import HuggingfaceMetric


def hf_metrics():
    return list_evaluation_modules(module_type="metric", include_community=False)


class TestHuggingfaceMetric:
    def test_all(self):
        all = [
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
        metrics = hf_metrics()
        assert len(metrics) == 53
        assert metrics == all

    @pytest.mark.parametrize("name", hf_metrics())
    def test_metric(self, name):
        try:
            m = HuggingfaceMetric(name)
            m.compute([0, 1, 1], [1, 0, 1])
        except NotImplementedError:
            pass
