from typing import Type, Optional

from sqlalchemy import Column, String, create_engine, select, inspect
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from fastrepl.run.cache.base import BaseCache

Base = declarative_base()


class LLMCache(Base):  # type: ignore
    __tablename__ = "llm_cache"
    model = Column(String, primary_key=True)
    prompt = Column(String, primary_key=True)
    response = Column(String)


class SQLAlchemyCache(BaseCache):
    __slot__ = ["engine", "schema"]

    def __init__(
        self,
        engine: Engine,
        schema: Type[LLMCache] = LLMCache,
    ) -> None:
        self.engine = engine
        self.schema = schema

        if not inspect(self.engine).has_table(self.schema.__tablename__):
            self.schema.metadata.create_all(self.engine)

    def lookup(self, model: str, prompt: str) -> Optional[str]:
        stmt = (
            select(self.schema.response)
            .where(self.schema.model == model)
            .where(self.schema.prompt == prompt)
        )
        with Session(self.engine) as session:
            rows = session.execute(stmt).fetchall()
            if rows:
                return rows[0][0]
        return None

    def update(self, model: str, prompt: str, response: str) -> None:
        with Session(self.engine) as session, session.begin():
            session.merge(self.schema(model=model, prompt=prompt, response=response))

    def clear(self) -> None:
        with Session(self.engine) as session:
            session.query(self.schema).delete()
            session.commit()


class SQLiteCache(SQLAlchemyCache):
    def __init__(self, path=".fastrepl.db") -> None:
        engine = create_engine(f"sqlite:///{path}")
        super().__init__(engine)
