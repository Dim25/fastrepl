from typing import Tuple, Literal, TypedDict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoConfig

from fastrepl.eval.base import BaseMetaEvalNode


SENTENCE_ANSWER_SIMILARITY_METRICS = Literal["sas", "semantic_answer_similarity"]


class SASResult(TypedDict):
    top_1_sas: List[float]
    top_k_sas: List[float]
    pred_label_matrix: List[List[float]]


# Modified from https://github.com/deepset-ai/haystack/blob/da677003181c2a2c03d5714672444138caea6be6/haystack/modeling/evaluation/metrics.py#L392
class SemanticAnswerSimilarityMetric(BaseMetaEvalNode):
    __slot__ = ("model", "is_cross_encoder")

    def __init__(self, model_name_or_path: str, use_gpu=False):
        config = AutoConfig.from_pretrained(model_name_or_path)
        if config.architectures is not None:
            self.is_cross_encoder = any(
                arch.endswith("ForSequenceClassification")
                for arch in config.architectures
            )

        device = None if use_gpu else "cpu"
        self.model = self._load_model(model_name_or_path, device=device)

    def _load_model(self, model_name_or_path: str, device):
        if self.is_cross_encoder:
            return CrossEncoder(model_name_or_path, device=device)
        else:
            return SentenceTransformer(model_name_or_path, device=device)

    def compute(
        self, predictions: List[List[str]], references: List[List[str]], **kwargs
    ):
        if self.is_cross_encoder:
            return self._compute_cross_encoder(predictions, references, **kwargs)
        else:
            return self._compute_bi_encoder(predictions, references, **kwargs)

    def _compute_cross_encoder(
        self, predictions: List[List[str]], references: List[List[str]], **kwargs
    ) -> SASResult:
        top_1_sas: List[float] = []
        top_k_sas: List[float] = []
        pred_label_matrix: List[List[float]] = []
        lengths: List[Tuple[int, int]] = []

        grid = []
        for preds, labels in zip(predictions, references):
            for p in preds:
                for l in labels:
                    grid.append((p, l))
            lengths.append((len(preds), len(labels)))
        scores = self.model.predict(grid, **kwargs)

        current_position = 0
        for len_p, len_l in lengths:
            scores_window = scores[current_position : current_position + len_p * len_l]
            # Per predicted doc there are len_l entries comparing it to all len_l labels.
            # So to only consider the first doc we have to take the first len_l entries
            top_1_sas.append(np.max(scores_window[:len_l]))
            top_k_sas.append(np.max(scores_window))
            pred_label_matrix.append(scores_window.reshape(len_p, len_l).tolist())
            current_position += len_p * len_l

        return {
            "top_1_sas": top_1_sas,
            "top_k_sas": top_k_sas,
            "pred_label_matrix": pred_label_matrix,
        }

    def _compute_bi_encoder(
        self, predictions: List[List[str]], references: List[List[str]], **kwargs
    ) -> SASResult:
        top_1_sas: List[float] = []
        top_k_sas: List[float] = []
        pred_label_matrix: List[List[float]] = []
        lengths: List[Tuple[int, int]] = []

        # For Bi-encoders we can flatten predictions and labels into one list
        all_texts: List[str] = []
        for p, l in zip(predictions, references):  # type: ignore
            # TODO potentially exclude (near) exact matches from computations
            all_texts.extend(p)
            all_texts.extend(l)
            lengths.append((len(p), len(l)))
        # then compute embeddings
        embeddings = self.model.encode(all_texts, **kwargs)

        # then select which embeddings will be used for similarity computations
        current_position = 0
        for len_p, len_l in lengths:
            pred_embeddings = embeddings[current_position : current_position + len_p, :]
            current_position += len_p
            label_embeddings = embeddings[
                current_position : current_position + len_l, :
            ]
            current_position += len_l
            sims = cosine_similarity(pred_embeddings, label_embeddings)
            top_1_sas.append(np.max(sims[0, :]))
            top_k_sas.append(np.max(sims))
            pred_label_matrix.append(sims.tolist())

        return {
            "top_1_sas": top_1_sas,
            "top_k_sas": top_k_sas,
            "pred_label_matrix": pred_label_matrix,
        }
