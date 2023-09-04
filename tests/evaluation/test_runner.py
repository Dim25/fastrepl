import pytest

from fastrepl.utils import getenv


@pytest.mark.fastrepl
def test_api_base():
    import litellm

    assert getenv("LITELLM_PROXY_API_BASE", "") != ""
    assert litellm.api_base is not None
