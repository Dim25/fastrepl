## Working with Argilla

[Argilla](https://docs.argilla.io/en/latest) is an open-source data curation platform for LLMs.

```python
import argilla as rg

dataset_rg = rg.load(rg_name)
dataset_ds = dataset_rg.to_datasets()
```

You might want to remove unnecessary columns and rename the rest to something like `input` and `reference`.

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
