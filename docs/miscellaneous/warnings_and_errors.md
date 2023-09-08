# Warnings and Errors

## Warnings

### VerbosityBias
Labels you provided has inbalanced length. This may lead to verbosity bias.

### InvalidPrediction
Predicted label or number is not within given labels or within range. `fastrepl` will set prediction to `None` for this sample. 

### IncompletePrediction
When calculating metric, `prediction is None`. This `prediction-reference pair` will be skipped.

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
