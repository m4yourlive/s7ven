
import sys

from PySide6.QtWidgets import QApplication

from seven.core.logging import LoggerService
from seven.ui.main_window import SevenMainWindow


def main() -> int:
    logging_service = LoggerService()
    logging_service.start()

    log = logging_service.get_logger("seven.app")

    try:
        log.info("Starting Seven Desktop")

        app = QApplication(sys.argv)
        app.setApplicationName("Seven")
        app.setOrganizationName("Seven")

        window = SevenMainWindow()
        window.show()

        log.success("Seven Desktop started successfully")

        exit_code = app.exec()

        log.info("Seven Desktop shutting down")

        return exit_code

    except Exception:
        log.exception("Fatal application error")
        raise

    finally:
        logging_service.stop()


if __name__ == "__main__":
    sys.exit(main())
