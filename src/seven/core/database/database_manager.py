"""Database lifecycle and migration management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine

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

    @staticmethod
    def _get_alembic_config() -> Config:
        """Load the project's Alembic configuration."""
        project_root = Path(__file__).resolve().parents[4]
        config_path = project_root / "alembic.ini"

        if not config_path.is_file():
            raise FileNotFoundError(
                f"Alembic configuration not found: {config_path}"
            )

        return Config(str(config_path))


    @staticmethod
    def _run_upgrade(
        connection: Any,
        alembic_config: Config,
    ) -> None:
        """Execute Alembic upgrade using an existing connection."""
        alembic_config.attributes["connection"] = connection
        command.upgrade(alembic_config, "head")

    async def initialize(self) -> None:
        """Initialize the database and apply pending migrations."""
        if self._initialized:
            return

        await self._session_manager.initialize()

        try:
            alembic_config = self._get_alembic_config()
            alembic_config.attributes["database_config"] = self._config

            async with self.engine.connect() as connection:
                await connection.run_sync(
                    self._run_upgrade,
                    alembic_config,
                )

            self._initialized = True

        except Exception:
            await self._session_manager.dispose()
            raise


    async def dispose(self) -> None:
        """Release database resources."""
        await self._session_manager.dispose()
        self._initialized = False
