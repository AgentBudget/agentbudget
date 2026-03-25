# Streaming Support + CI Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add streaming response cost tracking for OpenAI and Anthropic, and fix the dev/CI environment so async tests always pass.

**Architecture:**
- `_patch.py` wraps streaming generators with thin wrappers that accumulate cost from the final chunk (OpenAI) or `message_delta` event (Anthropic), then record it to the active session after the consumer finishes iterating.
- `pyproject.toml` already lists `pytest-asyncio` in dev deps — the issue is the CI workflow only runs on release. We add a `ci.yml` PR workflow.
- No new external dependencies are introduced.

**Tech Stack:** Python 3.9+, openai SDK (Stream/AsyncStream), anthropic SDK (Stream[RawMessageStreamEvent]/AsyncStream), pytest-asyncio

---

## Task 1: Add PR CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Write the workflow file**

```yaml
name: CI

on:
  push:
    branches: ["main"]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest --tb=short -q
```

**Step 2: Verify locally that tests still pass**

```bash
.venv/bin/pytest --tb=short -q
```
Expected: `129 passed`

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add PR test workflow across Python 3.9-3.13"
```

---

## Task 2: Add streaming wrapper helpers to `_patch.py`

**Files:**
- Modify: `agentbudget/_patch.py`

The strategy:
- For **OpenAI**: wrap the `Stream`/`AsyncStream` generator. OpenAI sends token usage in the *last* chunk when `stream_options={"include_usage": True}` is set. We scan all chunks for a non-None `.usage` and record cost after iteration ends.
- For **Anthropic**: wrap `Stream[RawMessageStreamEvent]`. We collect `input_tokens` from the `message_start` event and `output_tokens` from the `message_delta` event, then record cost after iteration ends.
- If the SDK is not installed, the wrappers are never called — same pattern as existing sync patching.
- We detect streaming vs non-streaming by checking if the return value is an instance of `Stream`/`AsyncStream`.

**Step 1: Write the failing test first (see Task 3 — do Task 3 Step 1 before implementing)**

**Step 2: Add `_wrap_openai_stream` and `_wrap_openai_async_stream` to `_patch.py`**

Add after the existing `_wrap_async_method` function (line 54):

```python
def _wrap_openai_stream(stream: Any, get_session: Callable) -> Any:
    """Wrap a synchronous OpenAI Stream to record cost after iteration."""

    def _iter():
        model: Optional[str] = None
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None

        for chunk in stream:
            # Every chunk carries the model name
            if model is None:
                model = getattr(chunk, "model", None)
            # OpenAI puts usage on the final chunk (when stream_options={"include_usage": True})
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
            yield chunk

        # After the consumer finishes iterating, record the cost
        session = get_session()
        if session is not None and model and prompt_tokens is not None and completion_tokens is not None:
            try:
                session.wrap(_FakeLLMResult(model, prompt_tokens, completion_tokens))
            except Exception:
                logger.debug("Failed to track streaming cost", exc_info=True)
                raise

    return _iter()


async def _wrap_openai_async_stream(stream: Any, get_session: Callable) -> Any:
    """Wrap an async OpenAI Stream to record cost after iteration."""
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    async for chunk in stream:
        if model is None:
            model = getattr(chunk, "model", None)
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
        yield chunk

    session = get_session()
    if session is not None and model and prompt_tokens is not None and completion_tokens is not None:
        try:
            session.wrap(_FakeLLMResult(model, prompt_tokens, completion_tokens))
        except Exception:
            logger.debug("Failed to track streaming cost", exc_info=True)
            raise


def _wrap_anthropic_stream(stream: Any, get_session: Callable) -> Any:
    """Wrap a synchronous Anthropic stream to record cost after iteration."""

    def _iter():
        model: Optional[str] = None
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None

        for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "message_start":
                msg = getattr(event, "message", None)
                if msg is not None:
                    model = getattr(msg, "model", None)
                    usage = getattr(msg, "usage", None)
                    if usage is not None:
                        input_tokens = getattr(usage, "input_tokens", None)
            elif event_type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage is not None:
                    output_tokens = getattr(usage, "output_tokens", None)
            yield event

        session = get_session()
        if session is not None and model and input_tokens is not None and output_tokens is not None:
            try:
                session.wrap(_FakeAnthropicResult(model, input_tokens, output_tokens))
            except Exception:
                logger.debug("Failed to track Anthropic streaming cost", exc_info=True)
                raise

    return _iter()


async def _wrap_anthropic_async_stream(stream: Any, get_session: Callable) -> Any:
    """Wrap an async Anthropic stream to record cost after iteration."""
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    async for event in stream:
        event_type = getattr(event, "type", None)
        if event_type == "message_start":
            msg = getattr(event, "message", None)
            if msg is not None:
                model = getattr(msg, "model", None)
                usage = getattr(msg, "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "input_tokens", None)
        elif event_type == "message_delta":
            usage = getattr(event, "usage", None)
            if usage is not None:
                output_tokens = getattr(usage, "output_tokens", None)
        yield event

    session = get_session()
    if session is not None and model and input_tokens is not None and output_tokens is not None:
        try:
            session.wrap(_FakeAnthropicResult(model, input_tokens, output_tokens))
        except Exception:
            logger.debug("Failed to track Anthropic async streaming cost", exc_info=True)
            raise


class _FakeLLMResult:
    """Minimal response-like object for session.wrap() from streaming."""
    def __init__(self, model: str, prompt_tokens: int, completion_tokens: int):
        self.model = model
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


class _FakeAnthropicResult:
    """Minimal Anthropic response-like object for session.wrap() from streaming."""
    def __init__(self, model: str, input_tokens: int, output_tokens: int):
        self.model = model
        self.usage = _FakeAnthropicUsage(input_tokens, output_tokens)


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeAnthropicUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
```

**Step 3: Update `_wrap_method` to detect and delegate streaming responses**

Modify `_wrap_method` (sync) and `_wrap_async_method` (async) to check if the return value is a `Stream`/`AsyncStream` instance and delegate to the streaming wrappers:

```python
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
```

And the async version similarly checks for `AsyncStream`.

**Step 4: Run the streaming tests**

```bash
.venv/bin/pytest tests/test_streaming.py -v
```
Expected: all streaming tests pass.

**Step 5: Run the full test suite**

```bash
.venv/bin/pytest --tb=short -q
```
Expected: 129+ passed, 0 failed.

**Step 6: Commit**

```bash
git add agentbudget/_patch.py
git commit -m "feat: track costs for streaming OpenAI and Anthropic responses"
```

---

## Task 3: Add streaming tests

**Files:**
- Create: `tests/test_streaming.py`

**Step 1: Write the test file**

Tests cover:
1. OpenAI sync streaming — cost recorded after consuming all chunks
2. OpenAI sync streaming — no cost if usage chunk absent (older API default)
3. OpenAI async streaming — cost recorded after consuming all chunks
4. Anthropic sync streaming — cost from message_start + message_delta events
5. Anthropic async streaming — same
6. Streaming with BudgetExhausted — raises when budget exceeded mid-stream
7. Drop-in mode (`agentbudget.init()`) patches streaming transparently

See full test code in implementation below.

**Step 2: Run to verify they fail before implementation**

```bash
.venv/bin/pytest tests/test_streaming.py -v
```
Expected: ImportError or AttributeError on missing streaming wrappers.

---

## Notes on the Approach

- **No breaking changes.** Non-streaming code path is unchanged — the streaming check is a guard that only fires when the return type is `Stream`/`AsyncStream`.
- **No new dependencies.** `openai` and `anthropic` are already optional — streaming support is equally optional.
- **Generator passthrough.** The consumer gets back a regular generator, not the original `Stream` object. This is fine because the consumer only calls `for chunk in stream:` — but it means methods like `.response`, `.get_final_message()` etc. on `Stream` are not available. This is an acceptable trade-off for v1 streaming support (document in PR).
- **`stream_options={"include_usage": True}`** must be set by the caller for OpenAI to send usage in the stream. Without it, cost is not tracked (silently skipped, not an error). Document this.
