from pathlib import Path

import pytest

from seven.core.logging import LoggerConfig, LoggerService


@pytest.fixture
def logger_service(tmp_path: Path) -> LoggerService:
    config = LoggerConfig(
        level="DEBUG",
        console_enabled=False,
        file_enabled=True,
        log_directory=tmp_path,
        serialize=False,
    )

    return LoggerService(config)


def test_logger_creates_log_file(
    logger_service: LoggerService,
) -> None:
    logger_service.start()

    try:
        logger_service.get_logger("Test").info("Test log message")

        assert logger_service.is_running
        assert logger_service.log_directory is not None
    finally:
        logger_service.stop()


def test_logger_start_twice(
    logger_service: LoggerService,
) -> None:
    logger_service.start()
    logger_service.start()

    assert logger_service.is_running

    logger_service.stop()


def test_logger_invalid_level() -> None:
    service = LoggerService(
        LoggerConfig(level="INVALID")
    )

    with pytest.raises(ValueError):
        service.start()


def test_logger_requires_handler() -> None:
    service = LoggerService(
        LoggerConfig(
            console_enabled=False,
            file_enabled=False,
        )
    )

    with pytest.raises(ValueError):
        service.start()