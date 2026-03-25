"""Monkey-patching for automatic LLM cost tracking.

Patches OpenAI and Anthropic client methods so every API call
is automatically tracked without any code changes.

Streaming support
-----------------
When the patched method returns a streaming response (``openai.Stream`` or
``anthropic.Stream``), the return value is replaced with a thin generator
wrapper that:

1. Yields every chunk/event unchanged to the caller.
2. After the caller exhausts the generator, records the accumulated cost to
   the active session.

For OpenAI this relies on the caller setting
``stream_options={"include_usage": True}`` so that token usage appears on the
final chunk.  If no usage chunk is present, cost tracking is silently skipped.

For Anthropic, token counts are collected from the ``message_start`` event
(input tokens) and the ``message_delta`` event (output tokens).
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Iterator, AsyncIterator, Optional

logger = logging.getLogger("agentbudget.patch")

# Store original methods so we can unpatch cleanly
_originals: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Internal helper objects
# ---------------------------------------------------------------------------

class _FakeUsage:
    """OpenAI-style usage object reconstructed from stream chunks."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeAnthropicUsage:
    """Anthropic-style usage object reconstructed from stream events."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeLLMResult:
    """Minimal response-like object passed to session.wrap() for streaming."""

    def __init__(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.model = model
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


class _FakeAnthropicResult:
    """Minimal Anthropic response-like object for session.wrap() from streaming."""

    def __init__(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.model = model
        self.usage = _FakeAnthropicUsage(input_tokens, output_tokens)


# ---------------------------------------------------------------------------
# Streaming wrapper classes (public so they can be unit-tested directly)
# ---------------------------------------------------------------------------

class _OpenAIStreamWrapper:
    """Wraps a synchronous OpenAI ``Stream`` to record cost after iteration.

    Implements the full iterator *and* context-manager protocol so user code
    using either pattern works transparently:

        # for-loop pattern
        for chunk in client.chat.completions.create(stream=True, ...):
            ...

        # context-manager pattern (proper connection cleanup)
        with client.chat.completions.create(stream=True, ...) as stream:
            for chunk in stream:
                ...

    Cost is recorded after the consumer exhausts the iterator.  Requires the
    caller to set ``stream_options={"include_usage": True}`` so that OpenAI
    includes token counts on the final chunk.  Without it, cost tracking is
    silently skipped (no error, no partial cost).
    """

    def __init__(self, stream: Any, get_session: Callable) -> None:
        self._stream = stream
        self._get_session = get_session

    def __iter__(self) -> Iterator:
        model: Optional[str] = None
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None

        for chunk in self._stream:
            if model is None:
                model = getattr(chunk, "model", None)
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                pt = getattr(usage, "prompt_tokens", None)
                ct = getattr(usage, "completion_tokens", None)
                if pt is not None:
                    prompt_tokens = pt
                if ct is not None:
                    completion_tokens = ct
            yield chunk

        self._record_cost(model, prompt_tokens, completion_tokens)

    def _record_cost(
        self,
        model: Optional[str],
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
    ) -> None:
        session = self._get_session()
        if (
            session is not None
            and model
            and prompt_tokens is not None
            and completion_tokens is not None
        ):
            try:
                session.wrap(_FakeLLMResult(model, prompt_tokens, completion_tokens))
            except Exception:
                logger.debug("Failed to track OpenAI streaming cost", exc_info=True)
                raise

    def __enter__(self) -> "_OpenAIStreamWrapper":
        return self

    def __exit__(self, *args: Any) -> None:
        # Close the underlying stream if it supports it (connection cleanup)
        close = getattr(self._stream, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass


class _AsyncOpenAIStreamWrapper:
    """Wraps an async OpenAI ``AsyncStream`` with full async iterator and
    async context-manager support.

        # async for pattern
        async for chunk in await client.chat.completions.create(stream=True):
            ...

        # async with pattern
        async with await client.chat.completions.create(stream=True) as stream:
            async for chunk in stream:
                ...
    """

    def __init__(self, stream: Any, get_session: Callable) -> None:
        self._stream = stream
        self._get_session = get_session

    def __aiter__(self) -> "_AsyncOpenAIStreamWrapper":
        return self

    async def __anext__(self) -> Any:
        # Delegate to the underlying async iterator
        raise StopAsyncIteration  # overridden by __aiter__ + _iterate below

    async def _iterate(self) -> AsyncIterator:
        model: Optional[str] = None
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None

        async for chunk in self._stream:
            if model is None:
                model = getattr(chunk, "model", None)
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                pt = getattr(usage, "prompt_tokens", None)
                ct = getattr(usage, "completion_tokens", None)
                if pt is not None:
                    prompt_tokens = pt
                if ct is not None:
                    completion_tokens = ct
            yield chunk

        session = self._get_session()
        if (
            session is not None
            and model
            and prompt_tokens is not None
            and completion_tokens is not None
        ):
            try:
                session.wrap(_FakeLLMResult(model, prompt_tokens, completion_tokens))
            except Exception:
                logger.debug("Failed to track OpenAI async streaming cost", exc_info=True)
                raise

    def __aiter__(self) -> Any:  # type: ignore[misc]
        return self._iterate().__aiter__()

    async def __aenter__(self) -> "_AsyncOpenAIStreamWrapper":
        return self

    async def __aexit__(self, *args: Any) -> None:
        close = getattr(self._stream, "aclose", None) or getattr(self._stream, "close", None)
        if close is not None:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(close):
                    await close()
                else:
                    close()
            except Exception:
                pass


class _AnthropicStreamWrapper:
    """Wraps a synchronous Anthropic ``Stream[RawMessageStreamEvent]``.

    Collects ``input_tokens`` from the ``message_start`` event and
    ``output_tokens`` from the ``message_delta`` event, then records cost
    after the caller exhausts the iterator.

    Supports both the for-loop and context-manager usage patterns.
    """

    def __init__(self, stream: Any, get_session: Callable) -> None:
        self._stream = stream
        self._get_session = get_session

    def __iter__(self) -> Iterator:
        model: Optional[str] = None
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None

        for event in self._stream:
            event_type = getattr(event, "type", None)
            if event_type == "message_start":
                msg = getattr(event, "message", None)
                if msg is not None:
                    if model is None:
                        model = getattr(msg, "model", None)
                    usage = getattr(msg, "usage", None)
                    if usage is not None:
                        it = getattr(usage, "input_tokens", None)
                        if it is not None:
                            input_tokens = it
            elif event_type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage is not None:
                    ot = getattr(usage, "output_tokens", None)
                    if ot is not None:
                        output_tokens = ot
            yield event

        session = self._get_session()
        if (
            session is not None
            and model
            and input_tokens is not None
            and output_tokens is not None
        ):
            try:
                session.wrap(_FakeAnthropicResult(model, input_tokens, output_tokens))
            except Exception:
                logger.debug("Failed to track Anthropic streaming cost", exc_info=True)
                raise

    def __enter__(self) -> "_AnthropicStreamWrapper":
        return self

    def __exit__(self, *args: Any) -> None:
        close = getattr(self._stream, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass


class _AsyncAnthropicStreamWrapper:
    """Wraps an async Anthropic ``AsyncStream[RawMessageStreamEvent]``."""

    def __init__(self, stream: Any, get_session: Callable) -> None:
        self._stream = stream
        self._get_session = get_session

    async def _iterate(self) -> AsyncIterator:
        model: Optional[str] = None
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None

        async for event in self._stream:
            event_type = getattr(event, "type", None)
            if event_type == "message_start":
                msg = getattr(event, "message", None)
                if msg is not None:
                    if model is None:
                        model = getattr(msg, "model", None)
                    usage = getattr(msg, "usage", None)
                    if usage is not None:
                        it = getattr(usage, "input_tokens", None)
                        if it is not None:
                            input_tokens = it
            elif event_type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage is not None:
                    ot = getattr(usage, "output_tokens", None)
                    if ot is not None:
                        output_tokens = ot
            yield event

        session = self._get_session()
        if (
            session is not None
            and model
            and input_tokens is not None
            and output_tokens is not None
        ):
            try:
                session.wrap(_FakeAnthropicResult(model, input_tokens, output_tokens))
            except Exception:
                logger.debug("Failed to track Anthropic async streaming cost", exc_info=True)
                raise

    def __aiter__(self) -> Any:
        return self._iterate().__aiter__()

    async def __aenter__(self) -> "_AsyncAnthropicStreamWrapper":
        return self

    async def __aexit__(self, *args: Any) -> None:
        close = getattr(self._stream, "aclose", None) or getattr(self._stream, "close", None)
        if close is not None:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(close):
                    await close()
                else:
                    close()
            except Exception:
                pass


# Public aliases used in tests and by _wrap_method
def _wrap_openai_stream(stream: Any, get_session: Callable) -> "_OpenAIStreamWrapper":
    return _OpenAIStreamWrapper(stream, get_session)


def _wrap_openai_async_stream(stream: Any, get_session: Callable) -> "_AsyncOpenAIStreamWrapper":
    return _AsyncOpenAIStreamWrapper(stream, get_session)


def _wrap_anthropic_stream(stream: Any, get_session: Callable) -> "_AnthropicStreamWrapper":
    return _AnthropicStreamWrapper(stream, get_session)


def _wrap_anthropic_async_stream(stream: Any, get_session: Callable) -> "_AsyncAnthropicStreamWrapper":
    return _AsyncAnthropicStreamWrapper(stream, get_session)


# ---------------------------------------------------------------------------
# Sync/async method wrappers (patched onto SDK classes)
# ---------------------------------------------------------------------------

def _wrap_method(original: Callable, get_session: Callable) -> Callable:
    """Wrap a sync SDK method to auto-track costs (streaming and non-streaming)."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        response = original(*args, **kwargs)
        session = get_session()
        if session is None:
            return response

        # Detect OpenAI sync stream
        try:
            from openai import Stream as OpenAIStream
            if isinstance(response, OpenAIStream):
                return _wrap_openai_stream(response, get_session)
        except ImportError:
            pass

        # Detect Anthropic sync stream
        try:
            from anthropic import Stream as AnthropicStream
            if isinstance(response, AnthropicStream):
                return _wrap_anthropic_stream(response, get_session)
        except ImportError:
            pass

        # Non-streaming path
        try:
            session.wrap(response)
        except Exception:
            logger.debug("Failed to track cost for response", exc_info=True)
            raise
        return response

    wrapper._agentbudget_patched = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_async_method(original: Callable, get_session: Callable) -> Callable:
    """Wrap an async SDK method to auto-track costs (streaming and non-streaming)."""

    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        response = await original(*args, **kwargs)
        session = get_session()
        if session is None:
            return response

        # Detect OpenAI async stream
        try:
            from openai import AsyncStream as OpenAIAsyncStream
            if isinstance(response, OpenAIAsyncStream):
                return _wrap_openai_async_stream(response, get_session)
        except ImportError:
            pass

        # Detect Anthropic async stream
        try:
            from anthropic import AsyncStream as AnthropicAsyncStream
            if isinstance(response, AnthropicAsyncStream):
                return _wrap_anthropic_async_stream(response, get_session)
        except ImportError:
            pass

        # Non-streaming path
        try:
            session.wrap(response)
        except Exception:
            logger.debug("Failed to track cost for response", exc_info=True)
            raise
        return response

    wrapper._agentbudget_patched = True  # type: ignore[attr-defined]
    return wrapper


# ---------------------------------------------------------------------------
# Provider patching
# ---------------------------------------------------------------------------

def patch_openai(get_session: Callable) -> bool:
    """Patch OpenAI client to automatically track costs.

    Returns True if patching succeeded, False if openai is not installed.
    """
    try:
        from openai.resources.chat.completions import Completions
    except ImportError:
        logger.debug("openai not installed, skipping patch")
        return False

    if getattr(getattr(Completions, "create", None), "_agentbudget_patched", False):
        return True  # already patched

    _originals["openai.chat.completions.create"] = Completions.create
    Completions.create = _wrap_method(Completions.create, get_session)  # type: ignore[assignment]

    # Patch async if available
    if hasattr(Completions, "acreate"):
        _originals["openai.chat.completions.acreate"] = Completions.acreate
        Completions.acreate = _wrap_async_method(Completions.acreate, get_session)  # type: ignore[assignment]

    # Also patch the async completions class if it exists
    try:
        from openai.resources.chat.completions import AsyncCompletions

        if not getattr(getattr(AsyncCompletions, "create", None), "_agentbudget_patched", False):
            _originals["openai.async_chat.completions.create"] = AsyncCompletions.create
            AsyncCompletions.create = _wrap_async_method(AsyncCompletions.create, get_session)  # type: ignore[assignment]
    except ImportError:
        pass

    logger.debug("Patched OpenAI client")
    return True


def patch_anthropic(get_session: Callable) -> bool:
    """Patch Anthropic client to automatically track costs.

    Returns True if patching succeeded, False if anthropic is not installed.
    """
    try:
        from anthropic.resources.messages import Messages
    except ImportError:
        logger.debug("anthropic not installed, skipping patch")
        return False

    if getattr(getattr(Messages, "create", None), "_agentbudget_patched", False):
        return True  # already patched

    _originals["anthropic.messages.create"] = Messages.create
    Messages.create = _wrap_method(Messages.create, get_session)  # type: ignore[assignment]

    # Patch async messages
    try:
        from anthropic.resources.messages import AsyncMessages

        if not getattr(getattr(AsyncMessages, "create", None), "_agentbudget_patched", False):
            _originals["anthropic.async_messages.create"] = AsyncMessages.create
            AsyncMessages.create = _wrap_async_method(AsyncMessages.create, get_session)  # type: ignore[assignment]
    except ImportError:
        pass

    logger.debug("Patched Anthropic client")
    return True


def unpatch_all() -> None:
    """Restore all original methods."""
    for key, original in _originals.items():
        if key == "openai.chat.completions.create":
            try:
                from openai.resources.chat.completions import Completions
                Completions.create = original  # type: ignore[assignment]
            except ImportError:
                pass
        elif key == "openai.chat.completions.acreate":
            try:
                from openai.resources.chat.completions import Completions
                Completions.acreate = original  # type: ignore[assignment]
            except ImportError:
                pass
        elif key == "openai.async_chat.completions.create":
            try:
                from openai.resources.chat.completions import AsyncCompletions
                AsyncCompletions.create = original  # type: ignore[assignment]
            except ImportError:
                pass
        elif key == "anthropic.messages.create":
            try:
                from anthropic.resources.messages import Messages
                Messages.create = original  # type: ignore[assignment]
            except ImportError:
                pass
        elif key == "anthropic.async_messages.create":
            try:
                from anthropic.resources.messages import AsyncMessages
                AsyncMessages.create = original  # type: ignore[assignment]
            except ImportError:
                pass

    _originals.clear()
    logger.debug("Unpatched all methods")
