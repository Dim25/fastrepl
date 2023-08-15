from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine

# TODO: Implement simple disk cache. Need to ignore/remove stale cache (~ few days?)


class SQLAlchemyCache:
    def __init__(self, engine: Engine):
        self.engine = engine


class SQLiteCache(SQLAlchemyCache):
    def __init__(self, database_path: str = ".fastrepl.db"):
        engine = create_engine(f"sqlite:///{database_path}")
        super().__init__(engine)
