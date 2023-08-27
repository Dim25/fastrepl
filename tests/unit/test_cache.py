import pytest
from _pytest.fixtures import FixtureRequest

import json

import fastrepl


def get_sqlite_cache() -> fastrepl.cache.SQLiteCache:
    return fastrepl.cache.SQLiteCache(path=":memory:")


CACHE_OPTIONS = [
    get_sqlite_cache,
]


@pytest.fixture(autouse=True, params=CACHE_OPTIONS)
def set_cache_and_teardown(request: FixtureRequest):
    cache_instance = request.param
    fastrepl.llm_cache = cache_instance()

    # fmt: off
    if fastrepl.llm_cache: fastrepl.llm_cache.clear()
    yield
    if fastrepl.llm_cache: fastrepl.llm_cache.clear()
    # fmt: on


class TestCache:
    def test_basic(self):
        assert fastrepl.llm_cache is not None

        fastrepl.llm_cache.update("model", "prompt", "response")
        assert fastrepl.llm_cache.lookup("model", "prompt") == "response"
        assert fastrepl.llm_cache.lookup("model", "prompt2") is None

        fastrepl.llm_cache.clear()
        assert fastrepl.llm_cache.lookup("model", "prompt") is None

    def test_json(self):
        assert fastrepl.llm_cache is not None

        msgs = [{"role": "user", "text": "Hello, how are you?"}]
        res = {"text": "1"}
        fastrepl.llm_cache.update("model", json.dumps(msgs), json.dumps(res))
        assert fastrepl.llm_cache.lookup("model", json.dumps(msgs)) == json.dumps(res)

    def test_multiple(self):
        assert fastrepl.llm_cache is not None
        assert fastrepl.llm_cache.lookup("model1", "prompt") is None

        fastrepl.llm_cache.update("model1", "prompt", "response1")
        assert fastrepl.llm_cache.lookup("model1", "prompt") == "response1"

        fastrepl.llm_cache.update("model1", "prompt", "response2")
        assert fastrepl.llm_cache.lookup("model1", "prompt") == "response2"

        fastrepl.llm_cache.update("model1", "prompt", "response3")
        assert fastrepl.llm_cache.lookup("model1", "prompt") == "response3"

        assert fastrepl.llm_cache.lookup("model2", "prompt") is None

        fastrepl.llm_cache.update("model2", "prompt", "response4")
        assert fastrepl.llm_cache.lookup("model2", "prompt") == "response4"

        fastrepl.llm_cache.update("model2", "prompt", "response5")
        assert fastrepl.llm_cache.lookup("model2", "prompt") == "response5"

        fastrepl.llm_cache.update("model2", "prompt", "response6")
        assert fastrepl.llm_cache.lookup("model2", "prompt") == "response6"

        assert fastrepl.llm_cache.lookup("model1", "prompt") == "response3"
