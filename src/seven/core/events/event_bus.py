from collections import defaultdict
from collections.abc import Callable
from threading import RLock

from seven.core.events.event import Event

EventHandler = Callable[[Event], None]


class EventBus:
    """
    Synchronous event dispatcher for Seven.

    Supports:
    - Subscribe
    - Unsubscribe
    - Publish
    - Multiple subscribers
    - Base event subscriptions
    - Thread-safe subscriber registry
    """

    def __init__(self, logger=None):
        self._subscribers: dict[
            type[Event], list[EventHandler]
        ] = defaultdict(list)

        self._lock = RLock()
        self._logger = logger

    def subscribe(
        self,
        event_type: type[Event],
        handler: EventHandler,
    ) -> None:
        """Register a handler for an event type."""

        if not isinstance(event_type, type) or not issubclass(
            event_type, Event
        ):
            raise TypeError("event_type must inherit from Event")

        if not callable(handler):
            raise TypeError("handler must be callable")

        with self._lock:
            handlers = self._subscribers[event_type]

            if handler not in handlers:
                handlers.append(handler)

        if self._logger:
            self._logger.debug(
                "Handler subscribed to {}", event_type.__name__
            )

    def unsubscribe(
        self,
        event_type: type[Event],
        handler: EventHandler,
    ) -> bool:
        """Remove a registered handler."""

        with self._lock:
            handlers = self._subscribers.get(event_type)

            if not handlers or handler not in handlers:
                return False

            handlers.remove(handler)

            if not handlers:
                self._subscribers.pop(event_type, None)

        if self._logger:
            self._logger.debug(
                "Handler unsubscribed from {}", event_type.__name__
            )

        return True

    def publish(self, event: Event) -> int:
        """
        Dispatch an event to all matching handlers.

        Returns the number of handlers executed successfully.
        A failed handler will not prevent other handlers from running.
        """

        if not isinstance(event, Event):
            raise TypeError("Published object must inherit from Event")

        handlers: list[EventHandler] = []

        with self._lock:
            for event_type in type(event).__mro__:
                if issubclass(event_type, Event):
                    handlers.extend(
                        self._subscribers.get(event_type, ())
                    )

        successful = 0

        for handler in handlers:
            try:
                handler(event)
                successful += 1

            except Exception:
                if self._logger:
                    self._logger.exception(
                        "Event handler failed for {}",
                        type(event).__name__,
                    )

        if self._logger:
            self._logger.debug(
                "Event {} dispatched to {} handlers",
                type(event).__name__,
                successful,
            )

        return successful

    def clear(self, event_type: type[Event] | None = None) -> None:
        """Remove all handlers or handlers for a specific event."""

        with self._lock:
            if event_type is None:
                self._subscribers.clear()
            else:
                self._subscribers.pop(event_type, None)

    def subscriber_count(
        self,
        event_type: type[Event],
    ) -> int:
        """Return the number of handlers for an event type."""

        with self._lock:
            return len(self._subscribers.get(event_type, ()))
