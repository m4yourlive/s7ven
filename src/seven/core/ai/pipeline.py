
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
from seven.core.conversations.context_aware_personalizer import (
    ContextAwarePersonalizer,
)
from seven.core.conversations.context_injector import ContextInjector
from seven.core.conversations.message_service import MessageService
from seven.core.conversations.personalization_contracts import (
    PersonalizationCategory,
    PersonalizationRequest,
)
from seven.core.conversations.personalization_engine import (
    PersonalizationEngine,
)
from seven.core.conversations.response_style_personalizer import (
    ResponseStylePersonalizer,
)
from seven.core.conversations.user_profile_contracts import UserProfile


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

        self._personalization_engine = PersonalizationEngine()
        self._response_style_personalizer = ResponseStylePersonalizer()
        self._context_aware_personalizer = ContextAwarePersonalizer()

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
    def _validate_profile(
        user_profile: UserProfile | None,
    ) -> None:
        """Validate the optional structured user profile."""
        if user_profile is not None and not isinstance(
            user_profile,
            UserProfile,
        ):
            raise TypeError(
                "user_profile must be a UserProfile instance."
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

    def _build_personalization_context(
        self,
        *,
        user_message: str,
        user_profile: UserProfile,
        messages: tuple[AIMessage, ...],
        conversation: ConversationContext,
    ) -> str:
        """Build context-aware personalization instructions."""
        memory_context = "\n".join(
            item.content
            for item in messages
            if (
                item.role is MessageRole.SYSTEM
                and item.content != conversation.system_prompt
            )
        )

        conversation_context = "\n".join(
            f"{item.role.value}: {item.content}"
            for item in conversation.history
        )

        personalization_request = PersonalizationRequest(
            user_message=user_message,
            profile=user_profile,
            memory_context=memory_context,
            conversation_context=conversation_context,
        )

        initial_result = self._personalization_engine.personalize(
            personalization_request
        )

        if not initial_result.applied:
            return ""

        context_result = self._context_aware_personalizer.personalize(
            personalization_request,
            initial_result,
        )

        if not context_result.applied:
            return ""

        style_result = self._response_style_personalizer.personalize(
            context_result
        )

        sections: list[str] = []

        if style_result.applied:
            sections.append(style_result.context)

        domain_directives = tuple(
            directive
            for directive in context_result.directives
            if (
                directive.category
                is PersonalizationCategory.DOMAIN_CONTEXT
            )
        )

        if domain_directives:
            domain_lines = ["Domain context instructions:"]

            for directive in domain_directives:
                domain_lines.append(f"- {directive.instruction}")

            sections.append("\n".join(domain_lines))

        return "\n\n".join(sections)

    def _inject_personalization(
        self,
        *,
        messages: tuple[AIMessage, ...],
        user_message: str,
        user_profile: UserProfile | None,
        conversation: ConversationContext,
    ) -> tuple[AIMessage, ...]:
        """Insert personalization instructions into the AI request."""
        if user_profile is None:
            return messages

        personalization_context = self._build_personalization_context(
            user_message=user_message,
            user_profile=user_profile,
            messages=messages,
            conversation=conversation,
        )

        if not personalization_context:
            return messages

        personalization_message = AIMessage(
            role=MessageRole.SYSTEM,
            content=personalization_context,
        )

        insertion_index = next(
            (
                index
                for index, item in enumerate(messages)
                if item.role is not MessageRole.SYSTEM
            ),
            len(messages),
        )

        return (
            *messages[:insertion_index],
            personalization_message,
            *messages[insertion_index:],
        )

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
        user_profile: UserProfile | None = None,
    ) -> GenerationResponse:
        """Generate an AI response and optionally persist the exchange."""
        if not isinstance(message, str) or not message.strip():
            raise AIRequestError("User message cannot be empty.")

        self._validate_persistence(message_service, conversation_id)
        self._validate_context_injector(context_injector)
        self._validate_profile(user_profile)

        context = self._get_context(conversation_id)

        user_message = AIMessage(
            role=MessageRole.USER,
            content=message.strip(),
        )

        messages = await self._build_messages(
            message=user_message,
            context=context,
            context_injector=context_injector,
        )

        messages = self._inject_personalization(
            messages=messages,
            user_message=user_message.content,
            user_profile=user_profile,
            conversation=context,
        )

        request = self._request_factory.build(
            messages=messages,
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
        user_profile: UserProfile | None = None,
    ) -> AsyncIterator[str]:
        """Stream an AI response and optionally persist the exchange."""
        if not isinstance(message, str) or not message.strip():
            raise AIRequestError("User message cannot be empty.")

        self._validate_persistence(message_service, conversation_id)
        self._validate_context_injector(context_injector)
        self._validate_profile(user_profile)

        context = self._get_context(conversation_id)

        user_message = AIMessage(
            role=MessageRole.USER,
            content=message.strip(),
        )

        messages = await self._build_messages(
            message=user_message,
            context=context,
            context_injector=context_injector,
        )

        messages = self._inject_personalization(
            messages=messages,
            user_message=user_message.content,
            user_profile=user_profile,
            conversation=context,
        )

        request = self._request_factory.build(
            messages=messages,
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
