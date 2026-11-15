"""AI response pipeline for S7VEN."""

from seven.core.ai.engine import AIEngine
from seven.core.ai.exceptions import AIRequestError
from seven.core.ai.models import (
    AIMessage,
    GenerationResponse,
    MessageRole,
)
from seven.core.ai.request_factory import GenerationRequestFactory

from collections.abc import AsyncIterator

class AIResponsePipeline:
    """Coordinate message processing and AI response generation."""

    def __init__(
        self,
        engine: AIEngine,
        request_factory: GenerationRequestFactory,
    ) -> None:
        if not isinstance(engine, AIEngine):
            raise TypeError("engine must be an AIEngine instance.")

        if not isinstance(
            request_factory,
            GenerationRequestFactory,
        ):
            raise TypeError(
                "request_factory must be a GenerationRequestFactory instance."
            )

        self._engine = engine
        self._request_factory = request_factory

    async def generate(
        self,
        message: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResponse:
        """Generate an AI response from a user message."""

        if not isinstance(message, str) or not message.strip():
            raise AIRequestError("User message cannot be empty.")

        user_message = AIMessage(
            role=MessageRole.USER,
            content=message.strip(),
        )

        request = self._request_factory.build(
            messages=(user_message,),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return await self._engine.generate(request)

    async def stream(
        self,
        message: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream an AI response from a user message."""

        if not isinstance(message, str) or not message.strip():
            raise AIRequestError("User message cannot be empty.")

        user_message = AIMessage(
            role=MessageRole.USER,
            content=message.strip(),
        )

        request = self._request_factory.build(
            messages=(user_message,),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        async for chunk in self._engine.stream(request):
            yield chunk