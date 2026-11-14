"""Alembic migration environment."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

import seven.core.database.models  # noqa: F401
from seven.core.database import Base, DatabaseConfig

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_database_config() -> DatabaseConfig:
    """Resolve the database configuration."""
    database_config = config.attributes.get("database_config")

    if database_config is not None:
        return database_config

    database_url = config.attributes.get("database_url")

    if database_url is not None:
        rendered_url = (
            database_url.render_as_string(hide_password=False)
            if hasattr(database_url, "render_as_string")
            else str(database_url)
        )

        from sqlalchemy.engine import make_url

        parsed_url = make_url(rendered_url)

        return DatabaseConfig(
            data_directory=Path(parsed_url.database).parent,
            database_name=Path(parsed_url.database).name,
        )

    return DatabaseConfig()


def get_database_url() -> str:
    """Resolve the asynchronous database URL."""
    return get_database_config().database_url.render_as_string(
        hide_password=False
    )


def run_migrations_offline() -> None:
    """Run migrations without creating a database connection."""
    context.configure(
        url=get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Configure and execute migrations using a connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an asynchronous engine and run migrations."""
    database_config = get_database_config()

    database_config.database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    configuration = config.get_section(
        config.config_ini_section,
        {},
    )

    configuration["sqlalchemy.url"] = get_database_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    try:
        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations using an asynchronous database connection."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
