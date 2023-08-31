# Dealing with bias
It is very important to **properly handle bias throughout the evaluation process.** This section will cover the different types of bias that can occur and how `fastrepl` can help you deal with them.

## Model Bias
> `TL;DR` Use different models for generation and evaluation. Compare multiple models if possible.
### Problem
Different model has different bias. For example, [GPT-4 favors itself with a 10% higher win rate while Claude-v1 favors itself with a 25% higher win rate.](https://arxiv.org/pdf/2306.05685.pdf) Although the author did express some uncertainty:

> However, they also favor other models and GPT-3.5 does not favor itself. Due to limited data and small differences, our study cannot determine whether the models exhibit a self-enhancement bias. Conducting a controlled study is challenging because we cannot easily rephrase a response to fit the style of another model without changing the quality.

### Solution
`fastrepl` uses [`litellm`](https://github.com/BerriAI/litellm) under the hood, which allows you to use any model you want.

```python
eval = fastrepl.LLMChainOfThoughtClassifier(
    model="gpt-3.5-turbo", # can be any model that litellm supports
    ...
)
```

## Position Bias
> `TL;DR` Order of samples matters. Shuffle them or use consensus mechanism `fastrepl` provides.
### Problem
[LLM exhibits a propensity to favor certain positions over others.](https://arxiv.org/pdf/2306.05685.pdf) Mostly, they have bias toward the first.


### Solution
You can pass eaither `shuffle` or `consensus` to `position_debias_strategy`.

#### Shuffle
Everything is shuffled per `compute` if possible.


#### Consensus
`fastrepl` do maximum 2 call per `compute`.
```python
eval = fastrepl.LLMChainOfThoughtClassifier(
    position_debias_strategy="consensus" # default="shuffle"
    ...
)
```

If two result is not same, `fastrepl` will return `None`. This should be handled using `Human-Eval` later.

## Verbosity Bias
> `TL;DR` LLM love longer samples. Keep eye on the warnings `fastrepl` gives you.
### Problem
[LLM judge favors longer, verbose responses.](https://arxiv.org/pdf/2306.05685.pdf)

### Solution
For now, best we can do is to provide warnings.

For example:

```python
eval = fastrepl.LLMChainOfThoughtClassifier(
    labels={
        "POSITIVE": "This text expresses favorable, affirmative, or optimistic sentiments, conveying a sense of happiness, satisfaction, or positivity in its tone and content."",
        "NEGATIVE": "This text is bad.",
    }
)
```

Will result `UserWarning` mentioning `This may bias the model to prefer the longer one.`