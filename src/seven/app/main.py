
"""S7VEN desktop application entry point."""

import sys

from seven.core.di.container import SevenContainer
from seven.gui import WebViewApplication


def main() -> int:
    """Initialize dependencies and run the S7VEN desktop application."""
    container = SevenContainer()

    runtime_manager = container.runtime_manager()

    application = WebViewApplication(
        runtime_manager=runtime_manager,
    )

    application.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
