# Warnings and Errors

## Warnings

### VerbosityBias
Labels you provided has inbalanced length. This may lead to verbosity bias.

### InvalidPrediction
Predicted label or number is not within given labels or within range. `fastrepl` will set prediction to `None` for this sample. 

### IncompletePrediction

### CompletionTruncated

### UnknownLLMException

### FloatGrading
Predicted grade is float, not integer.

## Errors

### EmptyPipeline

### NoneReference

### TokenizeNotImplemented
