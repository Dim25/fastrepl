# Warnings and Errors

## Warnings

### VerbosityBias
Labels you provided has inbalanced length. This may lead to verbosity bias.

### InvalidPrediction
Predicted label or number is not within given labels or within range. `fastrepl` will set prediction to `None` for this sample. 

### IncompletePrediction
When calculating metric, `prediction is None`. This `prediction-reference pair` will be skipped.
`prediction is None` when LLM API call or [consensus](/guides/dealing_with_bias) failed. If want to manually fill out the value. Take a look at [human-eval](/guides/human_eval.ipynb).

### CompletionTruncated
LLM completion is truncated due to `max_tokens`. Note this can be happened even for `max_tokens=1` and maximum `logit_bias` for desired tokens.

### UnknownLLMException
We got unexpected error from `litellm` side. Please report this.

### FloatGrading
Predicted grade is float, not integer.

## Errors

### EmptyPipeline
Evaluator pipeline is empty.

### EmptyPredictions
When calculating metric, both `predictions` and `references` are empty.

### NoneReference
When calculating metric, some `referece is None`.

### TokenizeNotImplemented
You should't see this error. Please report this.
