import pytest
import baserun
from datasets import Dataset, load_dataset

import fastrepl


def eval_name(evaluator: str, model: str) -> str:
    return f"fastrepl_yelp_review_{evaluator}_{model}"


labels = {
    "FIVE_STARS": "given review is likely to be 5 stars",
    "FOUR_STARS": "given review is likely to be 4 stars",
    "THREE_STARS": "given review is likely to be 3 stars",
    "TWO_STARS": "given review is likely to be 2 stars",
    "ONE_STAR": "given review is likely to be 1 star",
}


def label2number(example):
    if example["prediction"] is None:
        return example

    example["prediction"] = {
        "FIVE_STARS": 5,
        "FOUR_STARS": 4,
        "THREE_STARS": 3,
        "TWO_STARS": 2,
        "ONE_STAR": 1,
    }[example["prediction"]]

    return example


def grade2number(example):
    if example["prediction"] is None:
        return example

    example["prediction"] = float(example["prediction"])

    return example


@pytest.fixture
def dataset() -> Dataset:
    dataset = load_dataset("yelp_review_full", split="test")
    dataset = dataset.shuffle(seed=8)
    dataset = dataset.select(range(30))
    dataset = dataset.rename_column("text", "input")
    dataset = dataset.map(
        lambda row: {"reference": row["label"] + 1, "input": row["input"]},
        remove_columns=["label"],
    )
    return dataset


@pytest.mark.parametrize(
    "model, position_debias_strategy",
    [
        ("gpt-3.5-turbo", "shuffle"),
    ],
)
@pytest.mark.fastrepl
def test_llm_classification_head(dataset, model, position_debias_strategy):
    eval = fastrepl.Evaluator(
        pipeline=[
            fastrepl.LLMClassificationHead(
                model=model,
                context="You will get a input text from Yelp review. Classify it using the labels.",
                labels=labels,
                position_debias_strategy=position_debias_strategy,
            )
        ]
    )

    result = fastrepl.LocalRunner(evaluator=eval, dataset=dataset).run()
    result = result.map(label2number)

    for p, r in zip(result["prediction"], result["reference"]):
        baserun.evals.match(eval_name("LLMClassificationHead", model), str(p), str(r))

    accuracy, mse, mae = (
        fastrepl.load_metric(name).compute(
            predictions=result["prediction"],
            references=result["reference"],
        )[name]
        for name in ("accuracy", "mse", "mae")
    )

    baserun.log("metrics", {"accuracy": accuracy, "mse": mse, "mae": mae})

    assert accuracy > 0.09
    assert mse < 5
    assert mae < 3


@pytest.mark.parametrize(
    "model, position_debias_strategy",
    [
        ("gpt-3.5-turbo", "shuffle"),
    ],
)
@pytest.mark.fastrepl
def test_llm_classification_head_cot(dataset, model, position_debias_strategy):
    eval = fastrepl.Evaluator(
        pipeline=[
            fastrepl.LLMClassificationHeadCOT(
                model=model,
                context="You will get a input text from Yelp review. Classify it using the labels.",
                labels=labels,
                position_debias_strategy=position_debias_strategy,
            )
        ]
    )

    result = fastrepl.LocalRunner(evaluator=eval, dataset=dataset).run()
    result = result.map(label2number)

    for p, r in zip(result["prediction"], result["reference"]):
        baserun.evals.match(
            eval_name("LLMClassificationHeadCOT", model), str(p), str(r)
        )

    accuracy, mse, mae = (
        fastrepl.load_metric(name).compute(
            predictions=result["prediction"],
            references=result["reference"],
        )[name]
        for name in ("accuracy", "mse", "mae")
    )

    baserun.log("metrics", {"accuracy": accuracy, "mse": mse, "mae": mae})

    assert accuracy > 0.09
    assert mse < 5
    assert mae < 3


@pytest.mark.parametrize(
    "model",
    [
        ("gpt-3.5-turbo"),
    ],
)
@pytest.mark.fastrepl
def test_llm_grading_head(dataset, model):
    eval = fastrepl.Evaluator(
        pipeline=[
            fastrepl.LLMGradingHead(
                model=model,
                context="You will get a input text from Yelp review. Grade user's satisfaction from 1 to 5.",
                number_from=1,
                number_to=5,
            )
        ]
    )

    result = fastrepl.LocalRunner(evaluator=eval, dataset=dataset).run()
    result = result.map(grade2number)

    for p, r in zip(result["prediction"], result["reference"]):
        baserun.evals.match(eval_name("LLMGradingHead", model), str(p), str(r))

    accuracy, mse, mae = (
        fastrepl.load_metric(name).compute(
            predictions=result["prediction"],
            references=result["reference"],
        )[name]
        for name in ("accuracy", "mse", "mae")
    )

    baserun.log("metrics", {"accuracy": accuracy, "mse": mse, "mae": mae})

    assert accuracy > 0.09
    assert mse < 5
    assert mae < 3


@pytest.mark.parametrize(
    "model",
    [
        ("gpt-3.5-turbo"),
    ],
)
@pytest.mark.fastrepl
def test_grading_head_cot(dataset, model):
    eval = fastrepl.Evaluator(
        pipeline=[
            fastrepl.LLMGradingHeadCOT(
                model=model,
                context="You will get a input text from Yelp review. Grade user's satisfaction in integer from 1 to 5.",
                number_from=1,
                number_to=5,
            )
        ]
    )

    result = fastrepl.LocalRunner(evaluator=eval, dataset=dataset).run()
    result = result.map(grade2number)

    for p, r in zip(result["prediction"], result["reference"]):
        baserun.evals.match(eval_name("LLMGradingHeadCOT", model), str(p), str(r))

    accuracy, mse, mae = (
        fastrepl.load_metric(name).compute(
            predictions=result["prediction"],
            references=result["reference"],
        )[name]
        for name in ("accuracy", "mse", "mae")
    )

    baserun.log("metrics", {"accuracy": accuracy, "mse": mse, "mae": mae})

    assert accuracy > 0.09
    assert mse < 5
    assert mae < 3
