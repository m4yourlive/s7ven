
"""S7VEN desktop application entry point."""

import sys

from seven.core.di.container import SevenContainer
from seven.gui import WebViewApplication
from seven.gui.bridge import FrontendBridge


def main() -> int:
    """Initialize and run S7VEN."""
    container = SevenContainer()
    runtime_manager = container.runtime_manager()

    bridge = FrontendBridge(
        voice_assistant_provider=getattr(
            container,
            "voice_assistant",
            None,
        ),
        voice_config_provider=lambda: container.config_manager().config.voice,
    )

    try:
        application = WebViewApplication(
            runtime_manager=runtime_manager,
            bridge=bridge,
        )
    except TypeError as exc:
        if "bridge" not in str(exc):
            raise

        application = WebViewApplication(
            runtime_manager=runtime_manager,
        )
        application.bridge = bridge

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
