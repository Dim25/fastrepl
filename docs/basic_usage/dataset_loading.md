# Dataset Loading

Every functions in `FastREPL` that requires dataset as input will accept [Huggingface Datasets](https://huggingface.co/docs/datasets/).


## From Dictionary
```python
from datasets import Dataset

ds = Dataset.from_dict({
    "question": ["q1", "q2"],
    "answer": ["a1", "a2"],
})
```

## From Huggingface Hub
```python
from datasets import load_dataset

ds = load_dataset("rotten_tomatoes")
```

## From Argilla
```python
import argilla as rg

dataset_rg = rg.load("fastrepl")
dataset_ds = dataset_rg.to_datasets()
```
