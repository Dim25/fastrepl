from typing import Optional, List, Any

import json
import pytest
import _pytest.terminal

from fastrepl.utils import getenv

run_url: Optional[str] = None


def report(data: Any) -> None:
    s = json.dumps(data)
    # NOTE: This will be later parsed by fastrepl
    print(f"__FASTREPL_START_{s}_FASTREPL_END__")


def set_proxy():
    import litellm

    # NOTE: This will be provided in Github App
    api_base = getenv("LITELLM_PROXY_API_BASE", "")
    litellm.api_base = api_base if api_base != "" else None


@pytest.hookimpl(tryfirst=True)
def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--fastrepl",
        action="store_true",
        default=False,
        help="Enable experimental fastrepl evaluation runner",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "fastrepl: exepeiemental fastrepl testing")


def pytest_sessionstart(session: pytest.Session):
    if session.config.getoption("--fastrepl"):
        set_proxy()


def pytest_sessionfinish(session: pytest.Session):
    global run_url
    if session.config.getoption("--fastrepl"):
        run_url = "https://docs.fastrepl.com"


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]):
    if not config.getoption("--fastrepl"):
        for item in items:
            for marker in item.iter_markers():
                if marker.name == "fastrepl":
                    item.add_marker(
                        pytest.mark.skip(
                            "--fastrepl is not specified, skipping tests with fastrepl marker"
                        )
                    )


def pytest_terminal_summary(terminalreporter: _pytest.terminal.TerminalReporter):
    global run_url
    if run_url:
        terminalreporter.write_sep("=", "fastrepl summary", purple=True)
        terminalreporter.write_line(f"{run_url}")
