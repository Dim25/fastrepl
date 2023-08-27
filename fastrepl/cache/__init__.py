from typing import Optional

from fastrepl.cache.base import BaseCache
from fastrepl.cache.disk import SQLiteCache
from fastrepl.cache.remote import RemoteCache

llm_cache: Optional[BaseCache] = None
