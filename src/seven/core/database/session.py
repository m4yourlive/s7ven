"""Asynchronous SQLAlchemy database session management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from seven.core.database.database_config import DatabaseConfig


class DatabaseSessionManager:
    """Manage the asynchronous database engine and sessions."""

    def __init__(
        self,
        config: DatabaseConfig,
        *,
        echo: bool = False,
    ) -> None:
        """Initialize the database session manager."""
        self._config = config
        self._engine: AsyncEngine = create_async_engine(
            config.database_url,
            echo=echo,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        self._initialized = False

    @property
    def engine(self) -> AsyncEngine:
        """Return the managed asynchronous engine."""
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Return the asynchronous session factory."""
        return self._session_factory

    @property
    def is_initialized(self) -> bool:
        """Return whether the database manager is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Prepare the database directory and verify connectivity."""
        if self._initialized:
            return

        self._config.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        async with self._engine.connect() as connection:
            await connection.run_sync(lambda sync_connection: None)

        self._initialized = True

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Provide a managed database session.

        Transactions are controlled explicitly by the caller.
        The session is always closed when the context exits.
        """
        if not self._initialized:
            raise RuntimeError(
                "Database session manager has not been initialized."
            )

        async with self._session_factory() as database_session:
            yield database_session

    async def dispose(self) -> None:
        """Dispose of the engine and release its resources."""
        await self._engine.dispose()
        self._initialized = False
