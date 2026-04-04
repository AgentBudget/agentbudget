"""Tests for streaming response cost tracking.

Covers:
- OpenAI sync streaming (usage in final chunk)
- OpenAI async streaming
- OpenAI streaming with no usage chunk (cost skipped with warning)
- Anthropic sync streaming (usage from message_start + message_delta events)
- Anthropic async streaming
- BudgetExhausted raised during streaming consumption
- Drop-in mode (agentbudget.init()) patches streaming transparently
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Iterator, AsyncIterator
from unittest import mock

import pytest

import agentbudget
from agentbudget import AgentBudget, BudgetExhausted
from agentbudget._patch import (
    _wrap_openai_stream,
    _wrap_openai_async_stream,
    _wrap_anthropic_stream,
    _wrap_anthropic_async_stream,
)


# ---------------------------------------------------------------------------
# Fake objects that mimic just enough of the OpenAI SDK
# ---------------------------------------------------------------------------

class FakeOpenAIUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeOpenAIChunk:
    """Mimics openai.types.chat.ChatCompletionChunk"""
    def __init__(self, model="gpt-4o", usage=None):
        self.model = model
        self.usage = usage


def _openai_chunks(model="gpt-4o", prompt_tokens=100, completion_tokens=50) -> Iterator:
    """Yield a few delta chunks then a final chunk with usage (stream_options=include_usage)."""
    yield FakeOpenAIChunk(model=model, usage=None)
    yield FakeOpenAIChunk(model=model, usage=None)
    yield FakeOpenAIChunk(
        model=model,
        usage=FakeOpenAIUsage(prompt_tokens, completion_tokens),
    )


def _openai_chunks_no_usage(model="gpt-4o") -> Iterator:
    """Stream without usage data (caller did not set stream_options)."""
    yield FakeOpenAIChunk(model=model, usage=None)
    yield FakeOpenAIChunk(model=model, usage=None)


# ---------------------------------------------------------------------------
# Fake objects that mimic just enough of the Anthropic SDK
# ---------------------------------------------------------------------------

class FakeAnthropicInputUsage:
    def __init__(self, input_tokens):
        self.input_tokens = input_tokens


class FakeAnthropicOutputUsage:
    def __init__(self, output_tokens):
        self.output_tokens = output_tokens


class FakeAnthropicMessage:
    def __init__(self, model, input_tokens):
        self.model = model
        self.usage = FakeAnthropicInputUsage(input_tokens)


class FakeMessageStartEvent:
    type = "message_start"
    def __init__(self, model, input_tokens):
        self.message = FakeAnthropicMessage(model, input_tokens)


class FakeContentDeltaEvent:
    type = "content_block_delta"


class FakeMessageDeltaEvent:
    type = "message_delta"
    def __init__(self, output_tokens):
        self.usage = FakeAnthropicOutputUsage(output_tokens)


class FakeMessageStopEvent:
    type = "message_stop"


def _anthropic_events(
    model="claude-3-5-sonnet-20241022", input_tokens=100, output_tokens=50
):
    yield FakeMessageStartEvent(model, input_tokens)
    yield FakeContentDeltaEvent()
    yield FakeMessageDeltaEvent(output_tokens)
    yield FakeMessageStopEvent()


# ---------------------------------------------------------------------------
# OpenAI sync streaming tests
# ---------------------------------------------------------------------------

def test_openai_stream_records_cost():
    """Cost is recorded after the consumer finishes iterating."""
    budget = AgentBudget(max_spend="$5.00")
    with budget.session() as session:
        assert session.spent == 0.0

        stream = _wrap_openai_stream(
            _openai_chunks("gpt-4o", prompt_tokens=1000, completion_tokens=500),
            lambda: session,
        )
        chunks = list(stream)

    # gpt-4o: $2.50/M input, $10.00/M output
    # 1000 * 2.5e-6 + 500 * 10e-6 = 0.0025 + 0.005 = 0.0075
    assert len(chunks) == 3
    assert session.spent == pytest.approx(0.0075, rel=1e-4)


def test_openai_stream_no_usage_skips_cost():
    """If the stream has no usage chunk, cost is skipped and a warning is logged."""
    budget = AgentBudget(max_spend="$5.00")
    with budget.session() as session:
        stream = _wrap_openai_stream(_openai_chunks_no_usage(), lambda: session)
        chunks = list(stream)

    assert len(chunks) == 2
    assert session.spent == 0.0


def test_openai_stream_break_after_usage_chunk_records_cost():
    """Breaking after the final usage chunk should still record cost via finally."""
    budget = AgentBudget(max_spend="$5.00")
    with budget.session() as session:
        stream = _wrap_openai_stream(
            _openai_chunks("gpt-4o", prompt_tokens=1000, completion_tokens=500),
            lambda: session,
        )

        seen = 0
        for chunk in stream:
            seen += 1
            if chunk.usage is not None:
                break

    assert seen == 3
    assert session.spent == pytest.approx(0.0075, rel=1e-4)


def test_openai_stream_no_usage_logs_warning(caplog):
    """A warning makes missing usage visible instead of silently under-counting."""
    budget = AgentBudget(max_spend="$5.00")
    with budget.session() as session:
        with caplog.at_level(logging.WARNING, logger="agentbudget.patch"):
            stream = _wrap_openai_stream(_openai_chunks_no_usage(), lambda: session)
            chunks = list(stream)

    assert len(chunks) == 2
    assert session.spent == 0.0
    assert "OpenAI streaming call ended without complete usage data" in caplog.text


def test_openai_stream_passthrough():
    """Every chunk yielded by the original stream is yielded by the wrapper."""
    budget = AgentBudget(max_spend="$5.00")
    with budget.session() as session:
        original_chunks = list(_openai_chunks())
        stream = _wrap_openai_stream(
            iter(original_chunks), lambda: session
        )
        wrapped_chunks = list(stream)

    assert wrapped_chunks == original_chunks


def test_openai_stream_no_session_is_noop():
    """When no session is active, streaming is fully transparent."""
    stream = _wrap_openai_stream(
        _openai_chunks("gpt-4o", 100, 50),
        lambda: None,  # no active session
    )
    chunks = list(stream)
    assert len(chunks) == 3


def test_openai_stream_budget_exhausted():
    """BudgetExhausted propagates when budget runs out during cost recording."""
    budget = AgentBudget(max_spend="$0.001")  # tiny budget
    with pytest.raises(BudgetExhausted):
        with budget.session() as session:
            stream = _wrap_openai_stream(
                _openai_chunks("gpt-4o", prompt_tokens=10000, completion_tokens=5000),
                lambda: session,
            )
            list(stream)  # consume; cost recorded at end


# ---------------------------------------------------------------------------
# OpenAI async streaming tests
# ---------------------------------------------------------------------------

async def _openai_async_chunks(model="gpt-4o", prompt_tokens=100, completion_tokens=50):
    for chunk in _openai_chunks(model, prompt_tokens, completion_tokens):
        yield chunk


@pytest.mark.asyncio
async def test_openai_async_stream_records_cost():
    budget = AgentBudget(max_spend="$5.00")
    async with budget.async_session() as session:
        stream = _wrap_openai_async_stream(
            _openai_async_chunks("gpt-4o", 1000, 500),
            lambda: session,
        )
        chunks = [chunk async for chunk in stream]

    assert len(chunks) == 3
    assert session.spent == pytest.approx(0.0075, rel=1e-4)


@pytest.mark.asyncio
async def test_openai_async_stream_no_usage_skips_cost():
    async def _no_usage():
        for chunk in _openai_chunks_no_usage():
            yield chunk

    budget = AgentBudget(max_spend="$5.00")
    async with budget.async_session() as session:
        stream = _wrap_openai_async_stream(_no_usage(), lambda: session)
        chunks = [chunk async for chunk in stream]

    assert session.spent == 0.0
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_openai_async_stream_break_after_usage_chunk_records_cost():
    budget = AgentBudget(max_spend="$5.00")
    async with budget.async_session() as session:
        stream = _wrap_openai_async_stream(
            _openai_async_chunks("gpt-4o", 1000, 500),
            lambda: session,
        )

        seen = 0
        async for chunk in stream:
            seen += 1
            if chunk.usage is not None:
                break

    assert seen == 3
    assert session.spent == pytest.approx(0.0075, rel=1e-4)


# ---------------------------------------------------------------------------
# Anthropic sync streaming tests
# ---------------------------------------------------------------------------

def test_anthropic_stream_records_cost():
    """Cost recorded from message_start (input) + message_delta (output) events."""
    budget = AgentBudget(max_spend="$5.00")
    with budget.session() as session:
        stream = _wrap_anthropic_stream(
            _anthropic_events("claude-3-5-sonnet-20241022", input_tokens=1000, output_tokens=500),
            lambda: session,
        )
        events = list(stream)

    # claude-3-5-sonnet: $3.00/M input, $15.00/M output
    # 1000 * 3e-6 + 500 * 15e-6 = 0.003 + 0.0075 = 0.0105
    assert len(events) == 4
    assert session.spent == pytest.approx(0.0105, rel=1e-4)


def test_anthropic_stream_passthrough():
    """All events are yielded unchanged."""
    original = list(_anthropic_events())
    budget = AgentBudget(max_spend="$5.00")
    with budget.session() as session:
        stream = _wrap_anthropic_stream(iter(original), lambda: session)
        wrapped = list(stream)
    assert wrapped == original


def test_anthropic_stream_no_session_is_noop():
    stream = _wrap_anthropic_stream(_anthropic_events(), lambda: None)
    events = list(stream)
    assert len(events) == 4


def test_anthropic_stream_break_after_usage_event_records_cost():
    """Anthropic can recover cost once both input and output usage were seen."""
    budget = AgentBudget(max_spend="$5.00")
    with budget.session() as session:
        stream = _wrap_anthropic_stream(
            _anthropic_events("claude-3-5-sonnet-20241022", input_tokens=1000, output_tokens=500),
            lambda: session,
        )

        seen = 0
        for event in stream:
            seen += 1
            if getattr(event, "type", None) == "message_delta":
                break

    assert seen == 3
    assert session.spent == pytest.approx(0.0105, rel=1e-4)


# ---------------------------------------------------------------------------
# Anthropic async streaming tests
# ---------------------------------------------------------------------------

async def _anthropic_async_events(
    model="claude-3-5-sonnet-20241022", input_tokens=100, output_tokens=50
):
    for event in _anthropic_events(model, input_tokens, output_tokens):
        yield event


@pytest.mark.asyncio
async def test_anthropic_async_stream_records_cost():
    budget = AgentBudget(max_spend="$5.00")
    async with budget.async_session() as session:
        stream = _wrap_anthropic_async_stream(
            _anthropic_async_events("claude-3-5-sonnet-20241022", 1000, 500),
            lambda: session,
        )
        events = [e async for e in stream]

    assert len(events) == 4
    assert session.spent == pytest.approx(0.0105, rel=1e-4)


@pytest.mark.asyncio
async def test_anthropic_async_stream_break_after_usage_event_records_cost():
    budget = AgentBudget(max_spend="$5.00")
    async with budget.async_session() as session:
        stream = _wrap_anthropic_async_stream(
            _anthropic_async_events("claude-3-5-sonnet-20241022", 1000, 500),
            lambda: session,
        )

        seen = 0
        async for event in stream:
            seen += 1
            if getattr(event, "type", None) == "message_delta":
                break

    assert seen == 3
    assert session.spent == pytest.approx(0.0105, rel=1e-4)


# ---------------------------------------------------------------------------
# Context manager protocol tests
# ---------------------------------------------------------------------------

def test_openai_stream_supports_context_manager():
    """Wrapper must support 'with create(stream=True) as stream:' pattern."""
    budget = AgentBudget(max_spend="$5.00")
    with budget.session() as session:
        wrapped = _wrap_openai_stream(
            _openai_chunks("gpt-4o", 1000, 500),
            lambda: session,
        )
        with wrapped as stream:
            chunks = list(stream)

    assert len(chunks) == 3
    assert session.spent == pytest.approx(0.0075, rel=1e-4)


@pytest.mark.asyncio
async def test_openai_async_stream_supports_context_manager():
    """Async wrapper must support 'async with create(stream=True) as stream:' pattern."""
    budget = AgentBudget(max_spend="$5.00")
    async with budget.async_session() as session:
        wrapped = _wrap_openai_async_stream(
            _openai_async_chunks("gpt-4o", 1000, 500),
            lambda: session,
        )
        async with wrapped as stream:
            chunks = [chunk async for chunk in stream]

    assert len(chunks) == 3
    assert session.spent == pytest.approx(0.0075, rel=1e-4)


def test_anthropic_stream_supports_context_manager():
    """Anthropic wrapper must support context manager protocol."""
    budget = AgentBudget(max_spend="$5.00")
    with budget.session() as session:
        wrapped = _wrap_anthropic_stream(
            _anthropic_events("claude-3-5-sonnet-20241022", 1000, 500),
            lambda: session,
        )
        with wrapped as stream:
            events = list(stream)

    assert len(events) == 4
    assert session.spent == pytest.approx(0.0105, rel=1e-4)


# ---------------------------------------------------------------------------
# close() interface tests
# ---------------------------------------------------------------------------

def test_openai_stream_wrapper_has_close():
    """_OpenAIStreamWrapper exposes close() matching Stream interface."""
    budget = AgentBudget(max_spend="$5.00")
    with budget.session() as session:
        close_called = []

        class FakeStreamWithClose:
            def __iter__(self):
                return iter([])
            def close(self):
                close_called.append(True)

        wrapped = _wrap_openai_stream(FakeStreamWithClose(), lambda: session)
        assert hasattr(wrapped, "close")
        wrapped.close()

    assert close_called == [True]


def test_anthropic_stream_wrapper_has_close():
    """_AnthropicStreamWrapper exposes close() matching Stream interface."""
    budget = AgentBudget(max_spend="$5.00")
    with budget.session() as session:
        close_called = []

        class FakeStreamWithClose:
            def __iter__(self):
                return iter([])
            def close(self):
                close_called.append(True)

        wrapped = _wrap_anthropic_stream(FakeStreamWithClose(), lambda: session)
        assert hasattr(wrapped, "close")
        wrapped.close()

    assert close_called == [True]


@pytest.mark.asyncio
async def test_openai_async_stream_wrapper_has_aclose():
    """_AsyncOpenAIStreamWrapper exposes aclose() matching AsyncStream.close() interface."""
    budget = AgentBudget(max_spend="$5.00")
    async with budget.async_session() as session:
        close_called = []

        class FakeAsyncStreamWithClose:
            def __aiter__(self):
                return self
            async def __anext__(self):
                raise StopAsyncIteration
            async def aclose(self):
                close_called.append(True)

        wrapped = _wrap_openai_async_stream(FakeAsyncStreamWithClose(), lambda: session)
        assert hasattr(wrapped, "aclose")
        await wrapped.aclose()

    assert close_called == [True]


@pytest.mark.asyncio
async def test_anthropic_async_stream_wrapper_has_aclose():
    """_AsyncAnthropicStreamWrapper exposes aclose() matching AsyncStream.close() interface."""
    budget = AgentBudget(max_spend="$5.00")
    async with budget.async_session() as session:
        close_called = []

        class FakeAsyncStreamWithClose:
            def __aiter__(self):
                return self
            async def __anext__(self):
                raise StopAsyncIteration
            async def aclose(self):
                close_called.append(True)

        wrapped = _wrap_anthropic_async_stream(FakeAsyncStreamWithClose(), lambda: session)
        assert hasattr(wrapped, "aclose")
        await wrapped.aclose()

    assert close_called == [True]


# ---------------------------------------------------------------------------
# Drop-in mode integration test
# ---------------------------------------------------------------------------

def test_dropin_patches_openai_streaming():
    """agentbudget.init() wraps streaming OpenAI calls; cost is tracked end-to-end."""
    import sys, types
    from unittest import mock

    # Build a fake openai module with a Completions class whose create()
    # returns a stream-like object (an instance of our fake Stream class).
    openai_mod = types.ModuleType("openai")
    resources_mod = types.ModuleType("openai.resources")
    chat_mod = types.ModuleType("openai.resources.chat")
    completions_mod = types.ModuleType("openai.resources.chat.completions")

    # The fake Stream type — used for isinstance detection in _wrap_method
    class FakeStream:
        def __iter__(self):
            return iter(_openai_chunks("gpt-4o", 1000, 500))

    openai_mod.Stream = FakeStream
    openai_mod.AsyncStream = type("AsyncStream", (), {})

    class FakeCompletions:
        @staticmethod
        def create(*args, **kwargs):
            return FakeStream()  # returns a streaming response

    completions_mod.Completions = FakeCompletions  # type: ignore[attr-defined]

    with mock.patch.dict(
        sys.modules,
        {
            "openai": openai_mod,
            "openai.resources": resources_mod,
            "openai.resources.chat": chat_mod,
            "openai.resources.chat.completions": completions_mod,
        },
    ):
        session = agentbudget.init(budget="$5.00")
        try:
            # Simulate a streaming call going through the patched method
            from agentbudget._patch import _wrap_method
            from agentbudget._global import _get_session
            patched_create = _wrap_method(FakeCompletions.create, _get_session)
            result = patched_create()

            # result should be an _OpenAIStreamWrapper, not the raw FakeStream
            from agentbudget._patch import _OpenAIStreamWrapper
            assert isinstance(result, _OpenAIStreamWrapper), (
                f"Expected _OpenAIStreamWrapper, got {type(result)}"
            )

            # Consuming the stream should record cost
            list(result)
            # gpt-4o: 1000 * 2.5e-6 + 500 * 10e-6 = 0.0075
            assert agentbudget.spent() == pytest.approx(0.0075, rel=1e-4)
        finally:
            agentbudget.teardown()


def test_wrap_method_injects_openai_stream_usage_option():
    """The OpenAI patch should request usage automatically for streaming calls."""
    captured_kwargs = {}

    class FakeStream:
        def __iter__(self):
            return iter(_openai_chunks("gpt-4o", 1000, 500))

    def fake_create(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return FakeStream()

    from agentbudget._patch import _wrap_method

    patched_create = _wrap_method(fake_create, lambda: None, provider="openai")
    patched_create(stream=True)

    assert captured_kwargs["stream_options"] == {"include_usage": True}


def test_wrap_method_preserves_explicit_openai_stream_usage_option():
    """Caller-provided include_usage should not be overwritten by the patch."""
    captured_kwargs = {}

    class FakeStream:
        def __iter__(self):
            return iter(_openai_chunks("gpt-4o", 1000, 500))

    def fake_create(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return FakeStream()

    from agentbudget._patch import _wrap_method

    patched_create = _wrap_method(fake_create, lambda: None, provider="openai")
    patched_create(stream=True, stream_options={"include_usage": False, "foo": "bar"})

    assert captured_kwargs["stream_options"] == {"include_usage": False, "foo": "bar"}


@pytest.mark.asyncio
async def test_wrap_async_method_injects_openai_stream_usage_option():
    """The async OpenAI patch should mirror the sync kwargs injection logic."""
    captured_kwargs = {}

    class FakeAsyncStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    async def fake_create(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return FakeAsyncStream()

    from agentbudget._patch import _wrap_async_method

    patched_create = _wrap_async_method(fake_create, lambda: None, provider="openai")
    await patched_create(stream=True)

    assert captured_kwargs["stream_options"] == {"include_usage": True}
