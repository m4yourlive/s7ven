
"""SQLite persistence for scheduled tasks and executions."""

import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import UUID

from seven.core.integrations.scheduler.models import (
    ScheduleType,
    ScheduledExecution,
    ScheduledTask,
    ScheduledTaskStatus,
)


class SQLiteScheduledTaskStore:
    """Persist scheduled tasks and execution history in SQLite."""

    def __init__(self, database_path: str | Path = ":memory:") -> None:
        self._database_path = str(database_path)
        self._connection: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        """Initialize the database and create required tables."""

        if self._connection is not None:
            return

        if self._database_path != ":memory:":
            Path(self._database_path).expanduser().parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._connection = sqlite3.connect(self._database_path)
        self._connection.row_factory = sqlite3.Row

        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                schedule_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                action TEXT NOT NULL,
                schedule_type TEXT NOT NULL,
                run_at TEXT NOT NULL,
                parameters TEXT NOT NULL,
                interval_seconds REAL,
                max_runs INTEGER,
                enabled INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scheduled_executions (
                execution_id TEXT PRIMARY KEY,
                schedule_id TEXT NOT NULL,
                scheduled_for TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                output TEXT,
                error TEXT,
                FOREIGN KEY(schedule_id)
                    REFERENCES scheduled_tasks(schedule_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_execution_schedule
            ON scheduled_executions(schedule_id);

            CREATE INDEX IF NOT EXISTS idx_execution_time
            ON scheduled_executions(scheduled_for);
            """
        )
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.commit()

    async def close(self) -> None:
        """Close the database connection."""

        if self._connection is not None:
            self._connection.close()
            self._connection = None

    async def save_task(self, task: ScheduledTask) -> None:
        """Insert or update a scheduled task."""

        self._require_initialized()

        if not isinstance(task, ScheduledTask):
            raise TypeError("task must be a ScheduledTask.")

        self._connection.execute(
            """
            INSERT INTO scheduled_tasks (
                schedule_id,
                name,
                action,
                schedule_type,
                run_at,
                parameters,
                interval_seconds,
                max_runs,
                enabled
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(schedule_id) DO UPDATE SET
                name = excluded.name,
                action = excluded.action,
                schedule_type = excluded.schedule_type,
                run_at = excluded.run_at,
                parameters = excluded.parameters,
                interval_seconds = excluded.interval_seconds,
                max_runs = excluded.max_runs,
                enabled = excluded.enabled
            """,
            (
                str(task.schedule_id),
                task.name,
                task.action,
                task.schedule_type.value,
                task.run_at.isoformat(),
                json.dumps(task.parameters),
                task.interval_seconds,
                task.max_runs,
                int(task.enabled),
            ),
        )
        self._connection.commit()

    async def get_task(self, schedule_id: UUID) -> ScheduledTask | None:
        """Retrieve a scheduled task by identifier."""

        self._require_initialized()
        self._validate_id(schedule_id)

        row = self._connection.execute(
            "SELECT * FROM scheduled_tasks WHERE schedule_id = ?",
            (str(schedule_id),),
        ).fetchone()

        if row is None:
            return None

        return self._deserialize_task(row)

    async def list_tasks(self) -> tuple[ScheduledTask, ...]:
        """Return all persisted scheduled tasks."""

        self._require_initialized()

        rows = self._connection.execute(
            "SELECT * FROM scheduled_tasks ORDER BY name, schedule_id"
        ).fetchall()

        return tuple(self._deserialize_task(row) for row in rows)

    async def delete_task(self, schedule_id: UUID) -> bool:
        """Delete a scheduled task and its execution history."""

        self._require_initialized()
        self._validate_id(schedule_id)

        self._connection.execute(
            "DELETE FROM scheduled_tasks WHERE schedule_id = ?",
            (str(schedule_id),),
        )
        deleted = self._connection.execute("SELECT changes()").fetchone()[0]
        self._connection.commit()

        return deleted > 0

    async def save_execution(
        self,
        execution: ScheduledExecution,
    ) -> None:
        """Insert or update an execution record."""

        self._require_initialized()

        if not isinstance(execution, ScheduledExecution):
            raise TypeError(
                "execution must be a ScheduledExecution."
            )

        output = json.dumps(execution.output)

        self._connection.execute(
            """
            INSERT INTO scheduled_executions (
                execution_id,
                schedule_id,
                scheduled_for,
                status,
                started_at,
                finished_at,
                output,
                error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(execution_id) DO UPDATE SET
                schedule_id = excluded.schedule_id,
                scheduled_for = excluded.scheduled_for,
                status = excluded.status,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                output = excluded.output,
                error = excluded.error
            """,
            (
                str(execution.execution_id),
                str(execution.schedule_id),
                execution.scheduled_for.isoformat(),
                execution.status.value,
                (
                    execution.started_at.isoformat()
                    if execution.started_at is not None
                    else None
                ),
                (
                    execution.finished_at.isoformat()
                    if execution.finished_at is not None
                    else None
                ),
                output,
                execution.error,
            ),
        )
        self._connection.commit()

    async def get_execution_history(
        self,
        schedule_id: UUID,
        limit: int | None = None,
    ) -> tuple[ScheduledExecution, ...]:
        """Retrieve execution history for a scheduled task."""

        self._require_initialized()
        self._validate_id(schedule_id)

        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError("limit must be an integer or None.")

            if limit <= 0:
                raise ValueError("limit must be positive.")

        query = """
            SELECT * FROM scheduled_executions
            WHERE schedule_id = ?
            ORDER BY scheduled_for DESC, execution_id DESC
        """
        parameters: tuple[Any, ...] = (str(schedule_id),)

        if limit is not None:
            query += " LIMIT ?"
            parameters += (limit,)

        rows = self._connection.execute(
            query,
            parameters,
        ).fetchall()

        return tuple(
            self._deserialize_execution(row)
            for row in rows
        )

    def _require_initialized(self) -> None:
        if self._connection is None:
            raise RuntimeError("Store has not been initialized.")

    @staticmethod
    def _validate_id(schedule_id: UUID) -> None:
        if not isinstance(schedule_id, UUID):
            raise TypeError("schedule_id must be a UUID.")

    @staticmethod
    def _deserialize_task(row: sqlite3.Row) -> ScheduledTask:
        return ScheduledTask(
            name=row["name"],
            action=row["action"],
            schedule_type=ScheduleType(row["schedule_type"]),
            run_at=__import__("datetime").datetime.fromisoformat(
                row["run_at"]
            ),
            parameters=json.loads(row["parameters"]),
            interval_seconds=row["interval_seconds"],
            max_runs=row["max_runs"],
            enabled=bool(row["enabled"]),
            schedule_id=UUID(row["schedule_id"]),
        )

    @staticmethod
    def _deserialize_execution(
        row: sqlite3.Row,
    ) -> ScheduledExecution:
        from datetime import datetime

        return ScheduledExecution(
            schedule_id=UUID(row["schedule_id"]),
            scheduled_for=datetime.fromisoformat(row["scheduled_for"]),
            status=ScheduledTaskStatus(row["status"]),
            execution_id=UUID(row["execution_id"]),
            started_at=(
                datetime.fromisoformat(row["started_at"])
                if row["started_at"] is not None
                else None
            ),
            finished_at=(
                datetime.fromisoformat(row["finished_at"])
                if row["finished_at"] is not None
                else None
            ),
            output=json.loads(row["output"]),
            error=row["error"],
        )
