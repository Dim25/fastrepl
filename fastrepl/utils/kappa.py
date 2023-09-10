from typing import List, Any

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.inter_rater import cohens_kappa


def kappa(*predictions: List[Any]) -> float:
    if len(predictions) < 2:
        raise ValueError
    if len(predictions) > 2:
        raise NotImplementedError

    # TODO: We only support cohens_kappa for now
    assert len(predictions) == 2

    if len(predictions[0]) == 0 or len(predictions[1]) == 0:
        raise ValueError

    if isinstance(predictions[0][0], str):
        # TODO: workaround for None
        a = ["" if p is None else p for p in predictions[0]]
        b = ["" if p is None else p for p in predictions[1]]

        le = LabelEncoder()
        le.fit(list(set(a + b)))

        a, b = le.transform(a), le.transform(b)
    else:
        # TODO: workaround for None
        a = [-1 if p is None else p for p in predictions[0]]
        b = [-1 if p is None else p for p in predictions[1]]

    return cohens_kappa(table=confusion_matrix(a, b), return_results=False)
