
"""AI response pipeline for S7VEN."""

from collections.abc import AsyncIterator

from seven.core.ai.context_manager import (
    ConversationContext,
    ConversationContextManager,
)
from seven.core.ai.engine import AIEngine
from seven.core.ai.exceptions import AIRequestError
from seven.core.ai.models import (
    AIMessage,
    GenerationResponse,
    MessageRole,
)
from seven.core.ai.request_factory import GenerationRequestFactory
from seven.core.conversations.context_injector import ContextInjector
from seven.core.conversations.message_service import MessageService


class AIResponsePipeline:
    """Coordinate message processing and AI response generation."""

    def __init__(
        self,
        engine: AIEngine,
        request_factory: GenerationRequestFactory,
        context: ConversationContext | None = None,
        context_manager: ConversationContextManager | None = None,
    ) -> None:
        if not isinstance(engine, AIEngine):
            raise TypeError("engine must be an AIEngine instance.")

        if not isinstance(request_factory, GenerationRequestFactory):
            raise TypeError(
                "request_factory must be a GenerationRequestFactory instance."
            )

        if context is not None and not isinstance(
            context, ConversationContext
        ):
            raise TypeError(
                "context must be a ConversationContext instance."
            )

        if context_manager is not None and not isinstance(
            context_manager, ConversationContextManager
        ):
            raise TypeError(
                "context_manager must be a ConversationContextManager instance."
            )

        if context is not None and context_manager is not None:
            raise ValueError(
                "Provide either context or context_manager, not both."
            )

        self._engine = engine
        self._request_factory = request_factory
        self._context = context or ConversationContext()
        self._context_manager = context_manager

    @property
    def context(self) -> ConversationContext:
        """Return the default conversation context."""
        return self._context

    @property
    def context_manager(self) -> ConversationContextManager | None:
        """Return the multi-conversation context manager."""
        return self._context_manager

    def _get_context(
        self,
        conversation_id: str | None,
    ) -> ConversationContext:
        """Resolve the context for the current conversation."""
        if conversation_id is None:
            return self._context

        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError("conversation_id cannot be empty.")

        if self._context_manager is None:
            return self._context

        return self._context_manager.get_context(conversation_id)

    @staticmethod
    def _validate_persistence(
        message_service: MessageService | None,
        conversation_id: str | None,
    ) -> None:
        """Validate optional persistence dependencies."""
        if message_service is None and conversation_id is None:
            return

        if not isinstance(message_service, MessageService):
            raise TypeError(
                "message_service must be a MessageService instance."
            )

        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError(
                "A valid conversation_id is required for persistence."
            )

    @staticmethod
    def _validate_context_injector(
        context_injector: ContextInjector | None,
    ) -> None:
        """Validate the optional memory context injector."""
        if context_injector is not None and not isinstance(
            context_injector, ContextInjector
        ):
            raise TypeError(
                "context_injector must be a ContextInjector instance."
            )

    @staticmethod
    async def _build_messages(
        *,
        message: AIMessage,
        context: ConversationContext,
        context_injector: ContextInjector | None,
    ) -> tuple[AIMessage, ...]:
        """Build request history with optional memory injection."""
        history = context.build_messages()

        if context_injector is not None:
            history = await context_injector.inject(
                message.content,
                history=history,
            )

        return (*history, message)

    async def _persist_exchange(
        self,
        *,
        message_service: MessageService | None,
        conversation_id: str | None,
        user_message: AIMessage,
        assistant_content: str,
    ) -> None:
        """Persist a completed user-assistant exchange."""
        if message_service is None or conversation_id is None:
            return

        await self._persist_message(
            message_service,
            conversation_id,
            MessageRole.USER,
            user_message.content,
        )

        await self._persist_message(
            message_service,
            conversation_id,
            MessageRole.ASSISTANT,
            assistant_content,
        )

    @staticmethod
    async def _persist_message(
        message_service: MessageService,
        conversation_id: str,
        role: MessageRole,
        content: str,
    ) -> None:
        await message_service.create(
            conversation_id=conversation_id,
            role=role.value,
            content=content,
        )

    async def generate(
        self,
        message: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        message_service: MessageService | None = None,
        conversation_id: str | None = None,
        context_injector: ContextInjector | None = None,
    ) -> GenerationResponse:
        """Generate an AI response and optionally persist the exchange."""
        if not isinstance(message, str) or not message.strip():
            raise AIRequestError("User message cannot be empty.")

        self._validate_persistence(message_service, conversation_id)
        self._validate_context_injector(context_injector)

        context = self._get_context(conversation_id)

        user_message = AIMessage(
            role=MessageRole.USER,
            content=message.strip(),
        )

        request = self._request_factory.build(
            messages=await self._build_messages(
                message=user_message,
                context=context,
                context_injector=context_injector,
            ),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        response = await self._engine.generate(request)

        await self._persist_exchange(
            message_service=message_service,
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_content=response.content,
        )

        context.add_message(user_message)
        context.add_assistant_message(response.content)

        return response

    async def stream(
        self,
        message: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        message_service: MessageService | None = None,
        conversation_id: str | None = None,
        context_injector: ContextInjector | None = None,
    ) -> AsyncIterator[str]:
        """Stream an AI response and optionally persist the exchange."""
        if not isinstance(message, str) or not message.strip():
            raise AIRequestError("User message cannot be empty.")

        self._validate_persistence(message_service, conversation_id)
        self._validate_context_injector(context_injector)

        context = self._get_context(conversation_id)

        user_message = AIMessage(
            role=MessageRole.USER,
            content=message.strip(),
        )

        request = self._request_factory.build(
            messages=await self._build_messages(
                message=user_message,
                context=context,
                context_injector=context_injector,
            ),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        response_chunks: list[str] = []

        async for chunk in self._engine.stream(request):
            response_chunks.append(chunk)
            yield chunk

        response_content = "".join(response_chunks)

        if not response_content:
            return

        await self._persist_exchange(
            message_service=message_service,
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_content=response_content,
        )

        context.add_message(user_message)
        context.add_assistant_message(response_content)
