# Design Goals

## Single Import
`fastrepl` is designed to be used with a single `import fastrepl`. This requires more typing, but it makes it easier to learn and use.

### Non-interactive mode
This is for most use cases. In the normal python soure code, and tests.

```python
import fastrepl
```

### Interactive mode
This is intented to be used in a Jupyter notebook.

```python
import fastrepl.repl as fastrepl
```

## Framework Agnostic
`fastrepl` should work with any framework or setup. (It should be written in Python though. 😊)

