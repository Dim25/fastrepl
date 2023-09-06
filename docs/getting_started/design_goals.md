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
%env FASTREPL_INTERACTIVE=1
import fastrepl.repl as fastrepl
```

There is some code for [`prompt tuning`](https://github.com/fastrepl/fastrepl/blob/27bc80b60f1b5d035b5cbd96b3252054ceb3d241/fastrepl/repl/polish.py#L17) and [`visualization`](https://github.com/fastrepl/fastrepl/blob/27bc80b60f1b5d035b5cbd96b3252054ceb3d241/fastrepl/repl/context.py#L32), but it is not in a usable state yet.


## Framework Agnostic
It should be straightforward to use `fastrepl` with any framework or setup.
