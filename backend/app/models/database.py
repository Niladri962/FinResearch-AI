"""Engine and session factory. SQLite locally, PostgreSQL in production."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

SessionFactory = Callable[[], Session]


def build_engine(database_url: str, *, serverless: bool = False) -> Engine:
    if database_url.startswith("sqlite"):
        engine = create_engine(database_url, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

        return engine
    if serverless:
        # Function instances come and go; holding pooled connections would exhaust the
        # database. Use the provider's pooled connection string and open per use.
        return create_engine(database_url, poolclass=NullPool, pool_pre_ping=True)
    return create_engine(database_url, pool_pre_ping=True, pool_size=5, max_overflow=10)


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


@contextmanager
def session_scope(factory: SessionFactory) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on error, always close."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
