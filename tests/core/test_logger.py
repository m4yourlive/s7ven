
from pathlib import Path

import pytest

from seven.core.logging import LoggerConfig, LoggerService


def test_logger_creates_log_file(tmp_path: Path):
    config = LoggerConfig(
        console_enabled=False,
        file_enabled=True,
        serialize=False,
        log_directory=tmp_path,
    )

    service = LoggerService(config)
    service.start()

    log = service.get_logger("seven.test")
    log.info("Logger test successful")

    service.stop()

    log_file = tmp_path / "seven.log"

    assert log_file.exists()
    assert "Logger test successful" in log_file.read_text(
        encoding="utf-8"
    )


def test_logger_can_start_only_once(tmp_path: Path):
    config = LoggerConfig(
        console_enabled=False,
        file_enabled=True,
        serialize=False,
        log_directory=tmp_path,
    )

    service = LoggerService(config)

    service.start()
    service.start()

    assert service.is_running is True

    service.stop()


def test_logger_rejects_invalid_level():
    config = LoggerConfig(level="INVALID")
    service = LoggerService(config)

    with pytest.raises(ValueError):
        service.start()


def test_logger_requires_at_least_one_handler():
    config = LoggerConfig(
        console_enabled=False,
        file_enabled=False,
    )

    service = LoggerService(config)

    with pytest.raises(ValueError):
        service.start()
