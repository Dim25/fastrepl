**Most of the commands used for development are in `Taskfile.yaml`. [What is Taskfile?](https://taskfile.dev)**

## Setup
```bash
# This create .venv inside project.
# You might want to enable `Run Using Active Interpreter` to work with mypy vscode plugin
task install
```

## Test
```bash
task test
```

This will run tests that are marked with `@pytest.mark.todo`.
```bash
task test:todo
```

## TODO
You can find all the TODOs in the project by running

```bash
task todo
```

It requires [ripgrep](https://github.com/BurntSushi/ripgrep) to be installed.
