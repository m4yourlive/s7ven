"""Database lifecycle and schema initialization."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

from seven.core.database.base import Base
from seven.core.database.database_config import DatabaseConfig
from seven.core.database.session import DatabaseSessionManager


class DatabaseManager:
    """Coordinate database initialization and lifecycle."""

    def __init__(
        self,
        config: DatabaseConfig,
        *,
        echo: bool = False,
    ) -> None:
        """Initialize the database manager."""
        self._config = config
        self._session_manager = DatabaseSessionManager(
            config,
            echo=echo,
        )
        self._initialized = False

    @property
    def config(self) -> DatabaseConfig:
        """Return the database configuration."""
        return self._config

    @property
    def engine(self) -> AsyncEngine:
        """Return the managed database engine."""
        return self._session_manager.engine

    @property
    def session_manager(self) -> DatabaseSessionManager:
        """Return the database session manager."""
        return self._session_manager

    @property
    def is_initialized(self) -> bool:
        """Return whether the database is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the database and create missing tables."""
        if self._initialized:
            return

        await self._session_manager.initialize()

        try:
            async with self.engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)

            self._initialized = True

        except Exception:
            await self._session_manager.dispose()
            raise

    async def dispose(self) -> None:
        """Release database resources."""
        await self._session_manager.dispose()
        self._initialized = False
