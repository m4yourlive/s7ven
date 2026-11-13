"""JavaScript-Python bridge for the S7VEN frontend."""

from typing import Any


class FrontendBridge:
    """Expose safe Python methods to the JavaScript frontend."""

    def get_app_info(self) -> dict[str, str]:
        """Return basic application information."""
        return {
            "name": "S7VEN",
            "version": "0.1.0a0",
            "status": "ready",
        }

    def ping(self) -> dict[str, Any]:
        """Check whether the Python bridge is reachable."""
        return {
            "success": True,
            "message": "Python bridge is connected.",
        }

    def send_message(self, message: str) -> dict[str, Any]:
        """Receive and validate a message from the frontend."""
        if not isinstance(message, str):
            return {
                "success": False,
                "error": "Message must be a string.",
            }

        normalized_message = message.strip()

        if not normalized_message:
            return {
                "success": False,
                "error": "Message cannot be empty.",
            }

        return {
            "success": True,
            "message": normalized_message,
            "reply": (
                "Message received by S7VEN Core. "
                "AI response processing is not connected yet."
            ),
        }
