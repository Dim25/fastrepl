import pytest
from _pytest.fixtures import FixtureRequest

import json

import fastrepl
from fastrepl.run.cache import SQLiteCache


def get_sqlite_cache() -> SQLiteCache:
    return SQLiteCache(path=":memory:")


CACHE_OPTIONS = [
    get_sqlite_cache,
]


@pytest.fixture(autouse=True, params=CACHE_OPTIONS)
def set_cache_and_teardown(request: FixtureRequest):
    cache_instance = request.param
    fastrepl.cache = cache_instance()

    # fmt: off
    if fastrepl.cache: fastrepl.cache.clear()
    yield
    if fastrepl.cache: fastrepl.cache.clear()
    # fmt: on


class TestCache:
    def test_basic(self):
        assert fastrepl.cache is not None

        fastrepl.cache.update("model", "prompt", "response")
        assert fastrepl.cache.lookup("model", "prompt") == "response"
        assert fastrepl.cache.lookup("model", "prompt2") is None

        fastrepl.cache.clear()
        assert fastrepl.cache.lookup("model", "prompt") is None

    def test_json(self):
        assert fastrepl.cache is not None

        msgs = [{"role": "user", "text": "Hello, how are you?"}]
        res = {"text": "1"}
        fastrepl.cache.update("model", json.dumps(msgs), json.dumps(res))
        assert fastrepl.cache.lookup("model", json.dumps(msgs)) == json.dumps(res)

    def test_multiple(self):
        assert fastrepl.cache is not None
        assert fastrepl.cache.lookup("model1", "prompt") is None

        fastrepl.cache.update("model1", "prompt", "response1")
        assert fastrepl.cache.lookup("model1", "prompt") == "response1"

        fastrepl.cache.update("model1", "prompt", "response2")
        assert fastrepl.cache.lookup("model1", "prompt") == "response2"

        fastrepl.cache.update("model1", "prompt", "response3")
        assert fastrepl.cache.lookup("model1", "prompt") == "response3"

        assert fastrepl.cache.lookup("model2", "prompt") is None

        fastrepl.cache.update("model2", "prompt", "response4")
        assert fastrepl.cache.lookup("model2", "prompt") == "response4"

        fastrepl.cache.update("model2", "prompt", "response5")
        assert fastrepl.cache.lookup("model2", "prompt") == "response5"

        fastrepl.cache.update("model2", "prompt", "response6")
        assert fastrepl.cache.lookup("model2", "prompt") == "response6"

        assert fastrepl.cache.lookup("model1", "prompt") == "response3"
