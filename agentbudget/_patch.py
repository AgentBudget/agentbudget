"""Monkey-patching for automatic LLM cost tracking.

Patches OpenAI and Anthropic client methods so every API call
is automatically tracked without any code changes.

Streaming support
-----------------
When the patched method returns a streaming response (``openai.Stream`` or
``anthropic.Stream``), the return value is replaced with a wrapper object that:

1. Yields every chunk/event unchanged to the caller.
2. When iteration ends or the iterator is closed, records any accumulated cost
   to the active session.
3. Supports both the for-loop and context-manager usage patterns.

For OpenAI, the patch auto-injects
``stream_options={"include_usage": True}`` for streaming calls unless the
caller already set ``include_usage`` explicitly. OpenAI usage still arrives on
the final chunk, so tracking remains incomplete if that chunk is never
received.

For Anthropic, token counts are collected from the ``message_start`` event
(input tokens) and the ``message_delta`` event (output tokens).

Note: ``finally`` blocks let the wrappers record any usage already observed
before a consumer breaks out of the loop. They cannot recover usage the
provider never sent. If a stream ends without enough usage data to price it,
the patch logs a warning so the under-count is visible.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Mapping
from typing import Any, Callable, Iterator, AsyncIterator, Optional

logger = logging.getLogger("agentbudget.patch")

# Store original methods so we can unpatch cleanly
_originals: dict[str, Any] = {}


def _maybe_inject_openai_stream_options(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Ensure OpenAI streaming requests ask the API to include usage data.

    OpenAI exposes streaming usage on a final chunk when
    ``stream_options={"include_usage": True}`` is enabled. Injecting the flag
    here avoids a common footgun where tracking silently under-counts because
    the caller forgot that option.
    """
    if kwargs.get("stream") is not True:
        return kwargs

    stream_options = kwargs.get("stream_options")
    if stream_options is None:
        updated = dict(kwargs)
        updated["stream_options"] = {"include_usage": True}
        return updated

    if isinstance(stream_options, Mapping) and "include_usage" not in stream_options:
        updated = dict(kwargs)
        updated["stream_options"] = dict(stream_options)
        updated["stream_options"]["include_usage"] = True
        return updated

    return kwargs


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

    Cost is finalized in a ``finally`` block so breaking out of the loop still
    records usage that was already seen. OpenAI usage still arrives on the
    final chunk, so if that chunk never shows up the wrapper can only warn
    that tracking was incomplete.
    """

    def __init__(self, stream: Any, get_session: Callable) -> None:
        self._stream = stream
        self._get_session = get_session

    def __iter__(self) -> Iterator:
        model: Optional[str] = None
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None

        try:
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
        finally:
            self._record_cost(model, prompt_tokens, completion_tokens)

    def _record_cost(
        self,
        model: Optional[str],
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
    ) -> None:
        session = self._get_session()
        if session is None:
            return

        if model and prompt_tokens is not None and completion_tokens is not None:
            try:
                session.wrap(_FakeLLMResult(model, prompt_tokens, completion_tokens))
            except Exception:
                logger.debug("Failed to track OpenAI streaming cost", exc_info=True)
                raise
            return

        if model is not None or prompt_tokens is not None or completion_tokens is not None:
            logger.warning(
                "OpenAI streaming call ended without complete usage data; cost was not tracked."
            )

    def __enter__(self) -> "_OpenAIStreamWrapper":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying stream (matches Stream.close() interface)."""
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

    State is stored on ``self`` so that ``aclose()`` / ``__aexit__`` can
    record accumulated usage even when the caller breaks out of the loop
    early.  The old ``_iterate()`` async-generator approach caused the
    ``finally`` block to run inside a generator that was scheduled for
    collection rather than running synchronously, which lost the cost.
    """

    def __init__(self, stream: Any, get_session: Callable) -> None:
        self._stream = stream
        self._get_session = get_session
        self._aiter: Optional[Any] = None
        # Accumulated usage state
        self._model: Optional[str] = None
        self._prompt_tokens: Optional[int] = None
        self._completion_tokens: Optional[int] = None
        self._cost_recorded: bool = False

    def __aiter__(self) -> "_AsyncOpenAIStreamWrapper":
        self._aiter = self._stream.__aiter__()
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._aiter.__anext__()
        except StopAsyncIteration:
            self._record_cost()
            raise
        if self._model is None:
            self._model = getattr(chunk, "model", None)
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", None)
            ct = getattr(usage, "completion_tokens", None)
            if pt is not None:
                self._prompt_tokens = pt
            if ct is not None:
                self._completion_tokens = ct
        # Record eagerly once we have complete data so that a caller who
        # breaks immediately after seeing the usage chunk still gets tracked.
        if (
            self._model
            and self._prompt_tokens is not None
            and self._completion_tokens is not None
            and not self._cost_recorded
        ):
            self._record_cost()
        return chunk

    def _record_cost(self) -> None:
        if self._cost_recorded:
            return
        self._cost_recorded = True
        session = self._get_session()
        if session is None:
            return

        if (
            self._model
            and self._prompt_tokens is not None
            and self._completion_tokens is not None
        ):
            try:
                session.wrap(
                    _FakeLLMResult(self._model, self._prompt_tokens, self._completion_tokens)
                )
            except Exception:
                logger.debug("Failed to track OpenAI async streaming cost", exc_info=True)
                raise
            return

        if (
            self._model is not None
            or self._prompt_tokens is not None
            or self._completion_tokens is not None
        ):
            logger.warning(
                "OpenAI streaming call ended without complete usage data; cost was not tracked."
            )

    async def __aenter__(self) -> "_AsyncOpenAIStreamWrapper":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Record accumulated cost then close the underlying stream."""
        self._record_cost()
        close = getattr(self._stream, "aclose", None) or getattr(self._stream, "close", None)
        if close is not None:
            try:
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

        try:
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
        finally:
            self._record_cost(model, input_tokens, output_tokens)

    def _record_cost(
        self,
        model: Optional[str],
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> None:
        session = self._get_session()
        if session is None:
            return

        if model and input_tokens is not None and output_tokens is not None:
            try:
                session.wrap(_FakeAnthropicResult(model, input_tokens, output_tokens))
            except Exception:
                logger.debug("Failed to track Anthropic streaming cost", exc_info=True)
                raise
            return

        if model is not None or input_tokens is not None or output_tokens is not None:
            logger.warning(
                "Anthropic streaming call ended without complete usage data; cost was not tracked."
            )

    def __enter__(self) -> "_AnthropicStreamWrapper":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying stream (matches Stream.close() interface)."""
        close = getattr(self._stream, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass


class _AsyncAnthropicStreamWrapper:
    """Wraps an async Anthropic ``AsyncStream[RawMessageStreamEvent]``.

    Like ``_AsyncOpenAIStreamWrapper``, state is stored on ``self`` rather
    than inside an async generator so that ``aclose()`` can flush any
    accumulated cost when the caller breaks out of the loop early.
    """

    def __init__(self, stream: Any, get_session: Callable) -> None:
        self._stream = stream
        self._get_session = get_session
        self._aiter: Optional[Any] = None
        # Accumulated usage state
        self._model: Optional[str] = None
        self._input_tokens: Optional[int] = None
        self._output_tokens: Optional[int] = None
        self._cost_recorded: bool = False

    def __aiter__(self) -> "_AsyncAnthropicStreamWrapper":
        self._aiter = self._stream.__aiter__()
        return self

    async def __anext__(self) -> Any:
        try:
            event = await self._aiter.__anext__()
        except StopAsyncIteration:
            self._record_cost()
            raise
        event_type = getattr(event, "type", None)
        if event_type == "message_start":
            msg = getattr(event, "message", None)
            if msg is not None:
                if self._model is None:
                    self._model = getattr(msg, "model", None)
                usage = getattr(msg, "usage", None)
                if usage is not None:
                    it = getattr(usage, "input_tokens", None)
                    if it is not None:
                        self._input_tokens = it
        elif event_type == "message_delta":
            usage = getattr(event, "usage", None)
            if usage is not None:
                ot = getattr(usage, "output_tokens", None)
                if ot is not None:
                    self._output_tokens = ot
        # Record eagerly once both input and output usage are known so that a
        # caller who breaks after the message_delta event still gets tracked.
        if (
            self._model
            and self._input_tokens is not None
            and self._output_tokens is not None
            and not self._cost_recorded
        ):
            self._record_cost()
        return event

    def _record_cost(self) -> None:
        if self._cost_recorded:
            return
        self._cost_recorded = True
        session = self._get_session()
        if session is None:
            return

        if (
            self._model
            and self._input_tokens is not None
            and self._output_tokens is not None
        ):
            try:
                session.wrap(
                    _FakeAnthropicResult(self._model, self._input_tokens, self._output_tokens)
                )
            except Exception:
                logger.debug("Failed to track Anthropic async streaming cost", exc_info=True)
                raise
            return

        if (
            self._model is not None
            or self._input_tokens is not None
            or self._output_tokens is not None
        ):
            logger.warning(
                "Anthropic streaming call ended without complete usage data; cost was not tracked."
            )

    async def __aenter__(self) -> "_AsyncAnthropicStreamWrapper":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Record accumulated cost then close the underlying stream."""
        self._record_cost()
        close = getattr(self._stream, "aclose", None) or getattr(self._stream, "close", None)
        if close is not None:
            try:
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

def _wrap_method(
    original: Callable,
    get_session: Callable,
    provider: Optional[str] = None,
) -> Callable:
    """Wrap a sync SDK method to auto-track costs (streaming and non-streaming)."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if provider == "openai":
            kwargs = _maybe_inject_openai_stream_options(kwargs)
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


def _wrap_async_method(
    original: Callable,
    get_session: Callable,
    provider: Optional[str] = None,
) -> Callable:
    """Wrap an async SDK method to auto-track costs (streaming and non-streaming)."""

    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if provider == "openai":
            kwargs = _maybe_inject_openai_stream_options(kwargs)
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
    Completions.create = _wrap_method(Completions.create, get_session, provider="openai")  # type: ignore[assignment]

    # Patch async if available
    if hasattr(Completions, "acreate"):
        _originals["openai.chat.completions.acreate"] = Completions.acreate
        Completions.acreate = _wrap_async_method(Completions.acreate, get_session, provider="openai")  # type: ignore[assignment]

    # Also patch the async completions class if it exists
    try:
        from openai.resources.chat.completions import AsyncCompletions

        if not getattr(getattr(AsyncCompletions, "create", None), "_agentbudget_patched", False):
            _originals["openai.async_chat.completions.create"] = AsyncCompletions.create
            AsyncCompletions.create = _wrap_async_method(AsyncCompletions.create, get_session, provider="openai")  # type: ignore[assignment]
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
    Messages.create = _wrap_method(Messages.create, get_session, provider="anthropic")  # type: ignore[assignment]

    # Patch async messages
    try:
        from anthropic.resources.messages import AsyncMessages

        if not getattr(getattr(AsyncMessages, "create", None), "_agentbudget_patched", False):
            _originals["anthropic.async_messages.create"] = AsyncMessages.create
            AsyncMessages.create = _wrap_async_method(AsyncMessages.create, get_session, provider="anthropic")  # type: ignore[assignment]
    except ImportError:
        pass

    logger.debug("Patched Anthropic client")
    return True


def wrap_client(client: Any, session: Any) -> Any:
    """Attach a specific session to a specific client instance.

    Unlike ``agentbudget.init()`` which patches globally, this wraps a single
    client object so only calls on that instance are tracked. Useful when you
    have multiple clients, multiple budgets, or want explicit scope.

    Supported client types: ``openai.OpenAI``, ``openai.AsyncOpenAI``,
    ``anthropic.Anthropic``, ``anthropic.AsyncAnthropic``.

    Usage::

        budget = AgentBudget(max_spend="$5.00")
        with budget.session() as session:
            client = agentbudget.wrap_client(openai.OpenAI(), session)
            response = client.chat.completions.create(...)   # tracked
            other = openai.OpenAI()
            other.chat.completions.create(...)               # NOT tracked

    Returns the client with instance-level tracking attached (same object).
    """
    get_session = lambda: session  # noqa: E731

    # Detect client type by class name to avoid hard imports
    class_name = type(client).__name__

    if class_name in ("OpenAI", "AsyncOpenAI"):
        try:
            completions = client.chat.completions
            original_create = completions.__class__.create

            if class_name == "AsyncOpenAI":
                wrapped = _wrap_async_method(original_create, get_session, provider="openai")
            else:
                wrapped = _wrap_method(original_create, get_session, provider="openai")

            # Bind the wrapped method to this instance's completions object
            import types
            completions.create = types.MethodType(wrapped, completions)  # type: ignore[assignment]
        except AttributeError:
            logger.warning("wrap_client: could not attach to OpenAI client")

    elif class_name in ("Anthropic", "AsyncAnthropic"):
        try:
            messages = client.messages
            original_create = messages.__class__.create

            if class_name == "AsyncAnthropic":
                wrapped = _wrap_async_method(original_create, get_session, provider="anthropic")
            else:
                wrapped = _wrap_method(original_create, get_session, provider="anthropic")

            import types
            messages.create = types.MethodType(wrapped, messages)  # type: ignore[assignment]
        except AttributeError:
            logger.warning("wrap_client: could not attach to Anthropic client")

    else:
        logger.warning(
            "wrap_client: unrecognized client type %r — no tracking attached", class_name
        )

    return client


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
