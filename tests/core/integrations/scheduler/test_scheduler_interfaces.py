
from datetime import datetime, timezone
from uuid import UUID

from seven.core.integrations.scheduler.interfaces import (
    Scheduler,
    ScheduledTaskHandler,
    ScheduledTaskStore,
)
from seven.core.integrations.scheduler.models import (
    ScheduleType,
    ScheduledExecution,
    ScheduledTask,
)


class FakeScheduler:
    async def schedule(self, task: ScheduledTask) -> UUID:
        return task.schedule_id

    async def cancel(self, schedule_id: UUID) -> bool:
        return True

    async def get(self, schedule_id: UUID) -> ScheduledTask | None:
        return None


class FakeScheduledTaskHandler:
    async def execute(self, action: str, parameters: dict) -> object:
        return {"action": action}


class FakeScheduledTaskStore:
    async def save_task(self, task: ScheduledTask) -> None:
        pass

    async def get_task(self, schedule_id: UUID) -> ScheduledTask | None:
        return None

    async def list_tasks(self) -> tuple[ScheduledTask, ...]:
        return ()

    async def delete_task(self, schedule_id: UUID) -> None:
        pass

    async def save_execution(self, execution: ScheduledExecution) -> None:
        pass

    async def get_execution_history(
        self,
        schedule_id: UUID,
    ) -> tuple[ScheduledExecution, ...]:
        return ()


def test_scheduler_protocol_is_runtime_checkable():
    assert isinstance(FakeScheduler(), Scheduler)


def test_scheduled_task_handler_protocol_is_runtime_checkable():
    assert isinstance(FakeScheduledTaskHandler(), ScheduledTaskHandler)


def test_scheduled_task_store_protocol_is_runtime_checkable():
    assert isinstance(FakeScheduledTaskStore(), ScheduledTaskStore)


def test_scheduler_rejects_incomplete_implementation():
    class IncompleteScheduler:
        async def schedule(self, task):
            return task.schedule_id

    assert not isinstance(IncompleteScheduler(), Scheduler)


def test_task_handler_rejects_incomplete_implementation():
    class IncompleteHandler:
        async def execute(self, action, parameters):
            return None

    assert isinstance(IncompleteHandler(), ScheduledTaskHandler)


def test_store_rejects_incomplete_implementation():
    class IncompleteStore:
        async def save_task(self, task):
            pass

    assert not isinstance(IncompleteStore(), ScheduledTaskStore)


def test_scheduler_accepts_scheduled_task_contract():
    task = ScheduledTask(
        name="Daily backup",
        action="filesystem.backup",
        schedule_type=ScheduleType.ONCE,
        run_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
    )

    scheduler = FakeScheduler()

    assert isinstance(scheduler, Scheduler)
    assert task.schedule_id is not None
