# Dealing with bias
It is very important to **properly handle bias throughout the evaluation process.** This section will cover the different types of bias that can occur and how `fastrepl` can help you deal with them.

## Model Bias
> `TL;DR` Compare multiple models while doing evaluation
### Problem
Different model has different bias. For example, [GPT-4 favors itself with a 10% higher win rate while Claude-v1 favors itself with a 25% higher win rate.](https://arxiv.org/pdf/2306.05685.pdf) Although the author did express some uncertainty:

> However, they also favor other models and GPT-3.5 does not favor itself. Due to limited data and small differences, our study cannot determine whether the models exhibit a self-enhancement bias. Conducting a controlled study is challenging because we cannot easily rephrase a response to fit the style of another model without changing the quality.

Additionally, this tendency can be observed in [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/). When the evaluator is `GPT-4`, the ranking is as follows: `Claude 2 > ChatGPT > Claude`. However, when `Claude` serves as the evaluator, the ranking changes to `Claude > Claude 2 > ChatGPT`.

### Solution
`fastrepl` uses [`litellm`](https://github.com/BerriAI/litellm) under the hood, which allows you to use any model you want.

```python
eval = fastrepl.LLMClassificationHead(
    model="gpt-3.5-turbo", # can be any model that litellm supports
    ...
)
```

## Name Bias
> `TL;DR` Instead of the label names that the user provides, `fastrepl` will use an auto-generated mapping.

### Problem
[Claude-v1 also shows a name bias which makes it favors "Assistant A"](https://arxiv.org/pdf/2306.05685.pdf). This means that somehow, how we name a label can impact the evaluation results.

### Solution
When you are using `LLMClassificationHead`, `fastrepl` automatically generate single-token mapping for each label.

```python
eval = fastrepl.LLMClassificationHead(
    labels={
        "LABEL_1": "DESCRIPTION_1", # We ask LLM to output `A` for this
        "LABEL_2": "DESCRIPTION_2", # We ask LLM to output `B` for this
        # Of course, final result will be form of `LABEL_1`, not `A`.
    }
    ...
)
```

In the example above, the label name, such as `LABEL_1`, is provided solely for the user's convenience and is ignored by fastrepl. Instead, it maps them to `A` and `B` internally. Admittedly, this just removes the potential impact of the label name that the user provided, but it does have some advantages that make our evaluation more reliable, as we can use `logit_bias` to guide the LLM. In-depth research about the effectiveness of debiasing will be done in the future.

## Position Bias
> `TL;DR` Order of samples matters. Shuffle them or use consensus mechanism `fastrepl` provides.
### Problem
Position bias refers to the influence of the order of samples on the result. It is not unique to LLMs and has been observed in human decision-making and other machine learning domains.

We cannot be certain about which position the LLM prefers, as it varies among different LLMs and different prompts. For example, two papers that mention position bias show conflicting results:

> GPT-3.5: Biased toward first: 50.0%, Biased toward second: 1.2% (Table 2)
>
> \- [Judging LLM-as-a-judge with MT-Bench and Chatbot Arena](https://arxiv.org/pdf/2306.05685.pdf)


> This bias is also present in ChatGPT, which typically favors the second response.
>
> \- [Large Language Models are not Fair Evaluators](https://arxiv.org/pdf/2305.17926.pdf)


### Solution
You can pass eaither `shuffle` or `consensus` to `position_debias_strategy`.

#### Shuffle
[Label mapping](#name-bias) and order of samples are shuffled per `compute`.

#### Consensus
This is a simplified version of `Balanced Position Calibration`[(Large Language Models are not Fair Evaluators)](https://arxiv.org/pdf/2305.17926.pdf). `fastrepl` performs one additional prediction with the reversed sample ordering if the result of the first prediction was not in the exact middle.

```python
eval = fastrepl.LLMChainOfThoughtClassifier(
    position_debias_strategy="consensus" # default="shuffle"
    ...
)
```

If the two results are not the same, the evaluator will return `None`.

#### Others
The degree of position bias varies from one situation to another.

1. Position bias is less prominent when the score gap between the two responses becomes larger.

    For detailed results, refer to `Figure 2` of [Large Language Models are not Fair Evaluators](https://arxiv.org/pdf/2305.17926.pdf) and `Table 10` of [Judging LLM-as-a-judge with MT-Bench and Chatbot Arena](https://arxiv.org/pdf/2306.05685.pdf).

2. `Appendix D.1` of [Judging LLM-as-a-judge with MT-Bench and Chatbot Arena](https://arxiv.org/pdf/2306.05685.pdf) displays position bias across different models, prompts, and categories. For instance, position bias is much less prominent in math and coding categories.


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

This will result [VerbosityBiasWarning](miscellaneous/warnings_and_errors.md#verbositybias).
