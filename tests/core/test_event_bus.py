from dataclasses import dataclass

import pytest

from seven.core.events import Event, EventBus


@dataclass(frozen=True, slots=True)
class MessageReceived(Event):
    message: str


def test_publish_event() -> None:
    bus = EventBus()
    received = []

    def handler(event: MessageReceived) -> None:
        received.append(event.message)

    bus.subscribe(MessageReceived, handler)

    event = MessageReceived(message="Hello Seven")
    result = bus.publish(event)

    assert result == 1
    assert received == ["Hello Seven"]


def test_multiple_subscribers() -> None:
    bus = EventBus()
    results = []

    bus.subscribe(
        MessageReceived,
        lambda event: results.append("handler_1"),
    )

    bus.subscribe(
        MessageReceived,
        lambda event: results.append("handler_2"),
    )

    event = MessageReceived(message="Hello")
    result = bus.publish(event)

    assert result == 2
    assert len(results) == 2


def test_unsubscribe() -> None:
    bus = EventBus()
    received = []

    def handler(event: MessageReceived) -> None:
        received.append(event.message)

    bus.subscribe(MessageReceived, handler)

    assert bus.unsubscribe(MessageReceived, handler) is True

    bus.publish(MessageReceived(message="Hello"))

    assert received == []


def test_duplicate_subscription() -> None:
    bus = EventBus()

    def handler(event: MessageReceived) -> None:
        pass

    bus.subscribe(MessageReceived, handler)
    bus.subscribe(MessageReceived, handler)

    assert bus.subscriber_count(MessageReceived) == 1


def test_base_event_subscription() -> None:
    bus = EventBus()
    received = []

    bus.subscribe(Event, lambda event: received.append(event))

    event = MessageReceived(message="Hello")
    bus.publish(event)

    assert len(received) == 1
    assert received[0] is event


def test_invalid_event_type() -> None:
    bus = EventBus()

    with pytest.raises(TypeError):
        bus.subscribe(str, lambda event: None)


def test_invalid_published_object() -> None:
    bus = EventBus()

    with pytest.raises(TypeError):
        bus.publish("Not an event")


def test_failed_handler_does_not_stop_others() -> None:
    bus = EventBus()
    received = []

    def failing_handler(event: MessageReceived) -> None:
        raise RuntimeError("Handler error")

    def working_handler(event: MessageReceived) -> None:
        received.append(event.message)

    bus.subscribe(MessageReceived, failing_handler)
    bus.subscribe(MessageReceived, working_handler)

    result = bus.publish(
        MessageReceived(message="Still working")
    )

    assert result == 1
    assert received == ["Still working"]


def test_clear_subscribers() -> None:
    bus = EventBus()

    bus.subscribe(MessageReceived, lambda event: None)
    bus.clear(MessageReceived)

    assert bus.subscriber_count(MessageReceived) == 0
