"""S7VEN desktop application entry point."""

import sys

from seven.core.di.container import SevenContainer
from seven.gui import WebViewApplication


def main() -> int:
    """Initialize and run S7VEN."""
    container = SevenContainer()
    runtime_manager = container.runtime_manager()

    application = WebViewApplication(
        runtime_manager=runtime_manager,
    )

    create_tray_controller = getattr(
        application,
        "create_tray_controller",
        None,
    )

    if callable(create_tray_controller):
        create_tray_controller()

    application.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
