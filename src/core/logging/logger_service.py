import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from platformdirs import user_log_dir


@dataclass
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
    log_directory: Path | None = None


class LoggerService:
    """Centralized logging service for Seven."""

    def __init__(self, config: LoggerConfig | None = None) -> None:
        self.config = config or LoggerConfig()
        self._running = False
        self._log_directory: Path | None = None

    @property
    def is_running(self) -> bool:
        """Return whether the logging service is running."""
        return self._running

    @property
    def log_directory(self) -> Path | None:
        """Return the active log directory."""
        return self._log_directory

    def start(self) -> None:
        """Initialize logging handlers."""
        if self._running:
            return

        level = self.config.level.upper()

        if level not in logger._core.levels:
            raise ValueError(f"Invalid logging level: {level}")

        if not self.config.console_enabled and not self.config.file_enabled:
            raise ValueError(
                "At least one logging handler must be enabled."
            )

        logger.remove()

        if self.config.console_enabled:
            logger.add(
                sys.stderr,
                level=level,
                format=(
                    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>{extra[module]}</cyan> | "
                    "<level>{message}</level>"
                ),
                backtrace=self.config.backtrace,
                diagnose=self.config.diagnose,
            )

        if self.config.file_enabled:
            directory = self.config.log_directory or Path(
                user_log_dir(appname="Seven", appauthor="Seven")
            )

            directory.mkdir(parents=True, exist_ok=True)
            self._log_directory = directory

            logger.add(
                directory / "seven_{time:YYYY-MM-DD}.log",
                level=level,
                rotation=self.config.rotation,
                retention=self.config.retention,
                compression=self.config.compression,
                serialize=self.config.serialize,
                enqueue=True,
                backtrace=self.config.backtrace,
                diagnose=self.config.diagnose,
            )

        self._running = True

    def stop(self) -> None:
        """Remove logging handlers and stop the service."""
        if not self._running:
            return

        logger.remove()
        self._running = False

    def get_logger(self, module: str = "Seven"):
        """Return a module-bound logger instance."""
        return logger.bind(module=module)
