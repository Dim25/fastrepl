# Caching

- We have disk-based, exact matching caching enabled through `litellm`'s `GPTCache` integration.
```python
import fastrepl

fastrepl.LLMCache.enable()
# fastrepl.LLMCache.disable()
```

- We will add more caching options including remote caching in the future.
