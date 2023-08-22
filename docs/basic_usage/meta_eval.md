# Meta Evaluation
You **must also evaluate** how well your `automated evaluation` works, otherwise you might be optimizing for the wrong thing. Also, it can lower the evaluation cost, since cheaper model can be just as good as the expensive one.

## Prepare Dataset
To run meta-eval, you need human-labelled reference datasets. It might be convenient to use [Argilla](https://argilla.io) to manage them. You can always load fresh annotated dataset from Argilla.

`Figure 2: Human Evaluation Scores`[(LARGE LANGUAGE MODELS ON WIKIPEDIA-STYLE SURVEY GENERATION)](https://arxiv.org/pdf/2308.10410.pdf) is interesting. (High level of consensus)


## Run Evaluation
Run evaluation and grap `EvaluationResult`.

## Run Meta Evaluation
From `EvaluationResult`, you can get `MetaEvaluationResult`.
