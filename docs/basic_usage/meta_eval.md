# Meta Evaluation
You **must also evaluate** how well your `automated evaluation` works, otherwise you might be optimizing for the wrong thing. Also, it can lower the evaluation cost, since cheaper model can be just as good as the expensive one.

## Prepare Dataset
To run meta-eval, you need human-labelled reference datasets. It might be convenient to use [Argilla](https://argilla.io) to manage them. You can always load fresh annotated dataset from Argilla.

```python
import argilla as rg

dataset_rg = rg.load(rg_name)
dataset_ds = dataset_rg.to_datasets()
```

You might need to remove unnecessary columns and rename the rest to something like `input` and `reference`.

```python
input_ds = dataset_ds.remove_columns(
    [
        "inputs",
        "prediction",
        "prediction_agent",
        "annotation_agent",
        "vectors",
        "multi_label",
        "explanation",
        "id",
        "metadata",
        "status",
        "event_timestamp",
        "metrics",
    ]
)

assert len(input_ds.features) == 2
assert "text" in input_ds.features
assert "annotation" in input_ds.features

input_ds = input_ds.rename_column("text", "input")
input_ds = input_ds.rename_column("annotation", "reference")
```


## Run Evaluation
```python
from fastrepl.eval.model import LLMChainOfThought, LLMClassifier

result_ds = Evaluator(
    # Read more about Evaluator in docs/basic_usage/model_graded_eval
    ... 
).run()
```


## Run Meta Evaluation

```python
from fastrepl.eval.metric import load_metric

acc, mse, mae = load_metric("accuracy"), load_metric("mse"), load_metric("mae")

print(acc.compute(predictions=ds['prediction'], references=ds['reference']))
print(mse.compute(predictions=ds['prediction'], references=ds['reference']))
print(mae.compute(predictions=ds['prediction'], references=ds['reference']))
```
