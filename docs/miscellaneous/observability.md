# Observability

## Debug
`fastrepl.DEBUG` will use `DEBUG` env variable, and fallback to 0 if not set.

You can also set it manually at runtime using the following code:
```python
import fastrepl

fastrepl.DEBUG(0)
assert fastrepl.DEBUG == 0
```

- `DEBUG(0)`: No debugging
- `DEBUG(1)`: Will print a shortened LLM prompt.
- `DEBUG(2)`: Will print the entire LLM prompt.

## LiteLLM
Since `fastrepl` use `litellm` under the hood, you can do the following:
```python
import litellm

litellm.set_verbose = True
```
