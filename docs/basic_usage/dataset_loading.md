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

## From SQL
```python
from datasets import Dataset

# Fetch a database table
ds = Dataset.from_sql("test_data", "postgres:///db_name")

# Execute a SQL query on the table
ds = Dataset.from_sql("SELECT sentence FROM test_data", "postgres:///db_name")

# Use a Selectable object to specify the query
from sqlalchemy import select, text
stmt = select([text("sentence")]).select_from(text("test_data"))
ds = Dataset.from_sql(stmt, "postgres:///db_name")
```

## From Argilla
```python
import argilla as rg

dataset_rg = rg.load("fastrepl")
dataset_ds = dataset_rg.to_datasets()
```
