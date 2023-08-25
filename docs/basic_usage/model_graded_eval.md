# Model Graded Evaluation

## Choosing a Model
- Different models have different biases. So we need to try different models. You can use almost any models thanks to [LiteLLM](https://github.com/BerriAI/litellm).
- We should avoid using the same model for both the generation and evaluation. ([GPT-4 favors itself with a 10% higher win rate while Claude-v1 favors itself with a 25% higher win rate.](https://arxiv.org/abs/2306.05685))


- If both the generative model and the evaluation model are the same, they might share similar biases or errors, leading to a biased evaluation.

## `LLMChainOfThought` + `LLMClassifier`

## `LLMChainOfThoughtClassifier`
