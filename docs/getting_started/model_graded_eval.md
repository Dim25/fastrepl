# Model Graded Evaluation
There are some metric like [`rouge`](https://huggingface.co/spaces/evaluate-metric/rouge), but for most cases, especially in business, you'll need to evaluate more complex aspects of your model. 

## High level of consensus
> This Colab [notebook](https://colab.research.google.com/drive/1ctgygDRJhVGUJTQy8-bRZCl1WNcT8De6?usp=sharing) shows how to compute the agreement between humans and GPT-4 judge with the dataset. Our results show that humans and GPT-4 judge achieve over 80% agreement, the same level of agreement between humans.

\- [Large Model Systems Organization](https://lmsys.org/blog/2023-07-20-dataset/#agreement-calculation)

See `Figure 2: Human Evaluation Scores` in [Large language models on wikipedia-style survey generation](https://arxiv.org/pdf/2308.10410.pdf).

## TODO
[Style Over Substance: Evaluation Biases for Large Language Models](https://arxiv.org/abs/2307.03025)


Reference:
- https://eugeneyan.com/writing/llm-patterns/#evals-to-measure-performance