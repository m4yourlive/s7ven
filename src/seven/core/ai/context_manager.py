
"""Conversation context management for S7VEN."""

from collections import deque

from seven.core.ai.models import AIMessage, MessageRole


class ConversationContext:
    """Manage bounded conversation history for AI generation."""

    def __init__(
        self,
        system_prompt: str | None = None,
        max_messages: int = 20,
    ) -> None:
        if (
            not isinstance(max_messages, int)
            or isinstance(max_messages, bool)
            or max_messages < 1
        ):
            raise ValueError("max_messages must be a positive integer.")

        self._validate_system_prompt(system_prompt)

        self._system_prompt = system_prompt
        self._max_messages = max_messages
        self._history: deque[AIMessage] = deque(maxlen=max_messages)

    @staticmethod
    def _validate_system_prompt(prompt: str | None) -> None:
        if prompt is not None and not isinstance(prompt, str):
            raise TypeError("system_prompt must be a string or None.")

        if prompt is not None and not prompt.strip():
            raise ValueError("system_prompt cannot be empty.")

    @property
    def system_prompt(self) -> str | None:
        """Return the current system prompt."""
        return self._system_prompt

    @property
    def max_messages(self) -> int:
        """Return the maximum number of stored conversation messages."""
        return self._max_messages

    @property
    def history(self) -> tuple[AIMessage, ...]:
        """Return the stored conversation history."""
        return tuple(self._history)

    @property
    def message_count(self) -> int:
        """Return the number of stored conversation messages."""
        return len(self._history)

    def set_system_prompt(self, prompt: str | None) -> None:
        """Set or remove the system prompt."""
        self._validate_system_prompt(prompt)
        self._system_prompt = prompt

    def add_message(self, message: AIMessage) -> None:
        """Append a user or assistant message to the history."""
        if not isinstance(message, AIMessage):
            raise TypeError("message must be an AIMessage instance.")

        if message.role == MessageRole.SYSTEM:
            raise ValueError(
                "System messages must be managed through system_prompt."
            )

        self._history.append(message)

    def add_user_message(self, content: str) -> None:
        """Append a user message."""
        self.add_message(
            AIMessage(role=MessageRole.USER, content=content)
        )

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant message."""
        self.add_message(
            AIMessage(role=MessageRole.ASSISTANT, content=content)
        )

    def build_messages(self) -> tuple[AIMessage, ...]:
        """Build the message sequence for an AI generation request."""
        messages: list[AIMessage] = []

        if self._system_prompt is not None:
            messages.append(
                AIMessage(
                    role=MessageRole.SYSTEM,
                    content=self._system_prompt,
                )
            )

        messages.extend(self._history)
        return tuple(messages)

    def clear(self) -> None:
        """Clear conversation history while preserving the system prompt."""
        self._history.clear()


class ConversationContextManager:
    """Manage isolated contexts for multiple conversations."""

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        max_messages: int = 20,
    ) -> None:
        self._system_prompt = system_prompt
        self._max_messages = max_messages
        self._contexts: dict[str, ConversationContext] = {}

        # Validate configuration immediately.
        ConversationContext(
            system_prompt=system_prompt,
            max_messages=max_messages,
        )

    @staticmethod
    def _validate_conversation_id(conversation_id: str) -> None:
        if not isinstance(conversation_id, str):
            raise TypeError("conversation_id must be a string.")

        if not conversation_id.strip():
            raise ValueError("conversation_id cannot be empty.")

    @property
    def conversation_count(self) -> int:
        """Return the number of active conversation contexts."""
        return len(self._contexts)

    def get_context(self, conversation_id: str) -> ConversationContext:
        """Retrieve or create a context for a conversation."""
        self._validate_conversation_id(conversation_id)

        if conversation_id not in self._contexts:
            self._contexts[conversation_id] = ConversationContext(
                system_prompt=self._system_prompt,
                max_messages=self._max_messages,
            )

        return self._contexts[conversation_id]

    def remove_context(self, conversation_id: str) -> bool:
        """Remove a conversation context."""
        self._validate_conversation_id(conversation_id)

        return self._contexts.pop(conversation_id, None) is not None

    def clear_context(self, conversation_id: str) -> bool:
        """Clear a conversation's history without removing its context."""
        self._validate_conversation_id(conversation_id)

        context = self._contexts.get(conversation_id)

        if context is None:
            return False

        context.clear()
        return True

    def clear_all(self) -> None:
        """Remove all active conversation contexts."""
        self._contexts.clear()

    def set_system_prompt(self, prompt: str | None) -> None:
        """Update the default system prompt for future contexts."""
        ConversationContext._validate_system_prompt(prompt)
        self._system_prompt = prompt

        for context in self._contexts.values():
            context.set_system_prompt(prompt)
