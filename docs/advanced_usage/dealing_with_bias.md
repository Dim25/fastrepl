# Dealing with bias
It is very important to **properly handle bias throughout the evaluation process.** This section will cover the different types of bias that can occur and how `fastrepl` can help you deal with them.

## Model Bias
> `TL;DR` Use different models for generation and evaluation. Compare multiple models if possible.
### Problem

- [ GPT-4 favors itself with a 10% higher win rate while Claude-v1 favors itself with a 25% higher win rate.](https://arxiv.org/abs/2306.05685)

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
- [LLM exhibits a propensity to favor certain positions over others.](https://arxiv.org/pdf/2306.05685.pdf)

### Solution
#### Shuffle
Everything is shuffled, not only choices in the LLM prompt, but also for human evaluation.

#### Consensus
We have multiple consensus mechanisms, both human involved and not.


## Verbosity Bias
> `TL;DR` LLM love longer samples. Keep eye on the warnings `fastrepl` gives you.
### Problem
- [LLM judge favors longer, verbose responses.](https://arxiv.org/pdf/2306.05685.pdf)

### Solution
For now, best we can do is to provide warnings.

For example:

```python
# Code example goes here
```
