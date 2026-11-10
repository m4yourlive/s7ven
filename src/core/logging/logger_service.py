
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger
from platformdirs import user_log_dir


@dataclass(frozen=True, slots=True)
class LoggerConfig:
    """Configuration for the Seven logging service."""

    level: str = "DEBUG"
    console_enabled: bool = True
    file_enabled: bool = True

    rotation: str = "10 MB"
    retention: str = "14 days"
    compression: str = "zip"

    serialize: bool = True
    diagnose: bool = False
    backtrace: bool = False

    log_directory: Optional[Path] = None


class LoggerService:
    """
    Central logging service for Seven.

    Provides console and rotating file logging.
    Designed to be initialized once by the application runtime.
    """

    def __init__(self, config: Optional[LoggerConfig] = None):
        self.config = config or LoggerConfig()
        self._started = False
        self._handler_ids: list[int] = []
        self._log_directory: Optional[Path] = None

    @property
    def is_running(self) -> bool:
        return self._started

    @property
    def log_directory(self) -> Optional[Path]:
        return self._log_directory

    def start(self) -> None:
        """Initialize logging handlers."""

        if self._started:
            return

        self._validate_config()

        # Seven owns the global Loguru configuration.
        # Remove the default handler to avoid duplicate output.
        logger.remove()

        if self.config.console_enabled:
            console_id = logger.add(
                sys.stderr,
                level=self.config.level,
                colorize=True,
                format=(
                    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>{extra[module]}</cyan> | "
                    "<level>{message}</level>"
                ),
                diagnose=self.config.diagnose,
                backtrace=self.config.backtrace,
                enqueue=True,
            )

            self._handler_ids.append(console_id)

        if self.config.file_enabled:
            self._log_directory = self._resolve_log_directory()
            self._log_directory.mkdir(
                parents=True,
                exist_ok=True
            )

            file_path = self._log_directory / "seven.log"

            file_id = logger.add(
                str(file_path),
                level=self.config.level,
                rotation=self.config.rotation,
                retention=self.config.retention,
                compression=self.config.compression,
                serialize=self.config.serialize,
                encoding="utf-8",
                diagnose=self.config.diagnose,
                backtrace=self.config.backtrace,
                enqueue=True,
            )

            self._handler_ids.append(file_id)

        self._started = True

        self.get_logger("seven.logger").info(
            "Logger service initialized"
        )

    def stop(self) -> None:
        """Flush and remove logging handlers owned by Seven."""

        if not self._started:
            return

        for handler_id in self._handler_ids:
            logger.remove(handler_id)

        self._handler_ids.clear()
        self._started = False

    def get_logger(self, module: str):
        """Return a logger bound to a module name."""

        if not module.strip():
            raise ValueError("Logger module name cannot be empty.")

        return logger.bind(module=module)

    def _resolve_log_directory(self) -> Path:
        """Resolve a writable application log directory."""

        if self.config.log_directory is not None:
            return Path(self.config.log_directory).expanduser()

        return Path(
            user_log_dir(
                appname="Seven",
                appauthor="Seven"
            )
        )

    def _validate_config(self) -> None:
        """Validate logger configuration."""

        valid_levels = {
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        level = self.config.level.upper()

        if level not in valid_levels:
            raise ValueError(
                f"Invalid logging level: {self.config.level}"
            )

        if not self.config.console_enabled and not self.config.file_enabled:
            raise ValueError(
                "At least one logging handler must be enabled."
            )
