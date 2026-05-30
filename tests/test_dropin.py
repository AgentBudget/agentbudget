"""Tests for drop-in auto-instrumentation."""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from unittest import mock

import pytest

import agentbudget


class FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeResponse:
    def __init__(self, model="gpt-4o", prompt_tokens=100, completion_tokens=50):
        self.model = model
        self.usage = FakeUsage(prompt_tokens, completion_tokens)


def _build_fake_openai_modules():
    """Build a fake openai module tree that matches the SDK patch targets."""
    openai_mod = types.ModuleType("openai")
    resources_mod = types.ModuleType("openai.resources")
    chat_mod = types.ModuleType("openai.resources.chat")
    completions_mod = types.ModuleType("openai.resources.chat.completions")

    class Completions:
        def create(self, **kwargs):
            return FakeResponse(
                model=kwargs.get("model", "gpt-4o"),
                prompt_tokens=100,
                completion_tokens=50,
            )

    class AsyncCompletions:
        async def create(self, **kwargs):
            await asyncio.sleep(kwargs.get("delay", 0))
            return FakeResponse(
                model=kwargs.get("model", "gpt-4o"),
                prompt_tokens=100,
                completion_tokens=50,
            )

    completions_mod.Completions = Completions
    completions_mod.AsyncCompletions = AsyncCompletions
    chat_mod.completions = completions_mod
    resources_mod.chat = chat_mod
    openai_mod.resources = resources_mod

    return {
        "openai": openai_mod,
        "openai.resources": resources_mod,
        "openai.resources.chat": chat_mod,
        "openai.resources.chat.completions": completions_mod,
    }, Completions, AsyncCompletions


# ---- Tests for global API without patching ----

class TestGlobalAPI:
    def setup_method(self):
        agentbudget.teardown()

    def teardown_method(self):
        agentbudget.teardown()

    def test_init_returns_session(self):
        session = agentbudget.init(budget="$5.00")
        assert session is not None
        assert session.remaining == 5.0

    def test_spent_and_remaining(self):
        agentbudget.init(budget="$5.00")
        assert agentbudget.spent() == 0.0
        assert agentbudget.remaining() == 5.0

    def test_track_via_global(self):
        agentbudget.init(budget="$5.00")
        agentbudget.track(cost=0.50, tool_name="my_api")
        assert agentbudget.spent() == 0.50
        assert agentbudget.remaining() == 4.50

    def test_report_via_global(self):
        agentbudget.init(budget="$5.00")
        agentbudget.track(cost=1.0, tool_name="tool")
        r = agentbudget.report()
        assert r is not None
        assert r["total_spent"] == 1.0
        assert r["budget"] == 5.0

    def test_teardown_returns_report(self):
        agentbudget.init(budget="$5.00")
        agentbudget.track(cost=0.25, tool_name="api")
        r = agentbudget.teardown()
        assert r is not None
        assert r["total_spent"] == 0.25

    def test_teardown_clears_state(self):
        agentbudget.init(budget="$5.00")
        agentbudget.teardown()
        assert agentbudget.get_session() is None
        assert agentbudget.spent() == 0.0
        assert agentbudget.report() is None

    def test_double_init_tears_down_first(self):
        agentbudget.init(budget="$5.00")
        agentbudget.track(cost=1.0, tool_name="api")
        # Second init should start fresh
        agentbudget.init(budget="$10.00")
        assert agentbudget.spent() == 0.0
        assert agentbudget.remaining() == 10.0

    def test_track_without_init_raises(self):
        with pytest.raises(RuntimeError, match="init"):
            agentbudget.track(cost=0.01)

    def test_budget_enforcement(self):
        agentbudget.init(budget="$0.10")
        agentbudget.track(cost=0.05)
        with pytest.raises(agentbudget.BudgetExhausted):
            agentbudget.track(cost=0.06)

    def test_get_session_for_manual_use(self):
        agentbudget.init(budget="$5.00")
        session = agentbudget.get_session()
        assert session is not None
        # Can use session.wrap() manually alongside auto-tracking
        response = FakeResponse("gpt-4o", 100, 50)
        session.wrap(response)
        assert agentbudget.spent() > 0


# ---- Tests for monkey-patching mechanism ----

class TestPatching:
    def setup_method(self):
        agentbudget.teardown()

    def teardown_method(self):
        agentbudget.teardown()
        # Clean up fake provider modules
        for key in list(sys.modules.keys()):
            if key.startswith("openai") or key.startswith("anthropic"):
                del sys.modules[key]

    def _install_fake_openai(self):
        """Install a fake openai module that mimics the real SDK structure."""
        modules, Completions, _ = _build_fake_openai_modules()
        sys.modules.update(modules)

        return Completions

    def test_openai_patching(self):
        FakeCompletions = self._install_fake_openai()

        agentbudget.init(budget="$5.00")

        client = FakeCompletions()
        response = client.create(model="gpt-4o")

        assert response.model == "gpt-4o"
        assert agentbudget.spent() > 0

    def test_openai_unpatching(self):
        FakeCompletions = self._install_fake_openai()
        original_create = FakeCompletions.create

        agentbudget.init(budget="$5.00")
        assert FakeCompletions.create is not original_create  # patched

        agentbudget.teardown()
        assert FakeCompletions.create is original_create  # restored

    def test_no_tracking_after_teardown(self):
        FakeCompletions = self._install_fake_openai()

        agentbudget.init(budget="$5.00")
        agentbudget.teardown()

        # After teardown, calls should work but not be tracked
        client = FakeCompletions()
        response = client.create(model="gpt-4o")
        assert response.model == "gpt-4o"
        assert agentbudget.spent() == 0.0

    def test_budget_enforcement_through_patch(self):
        FakeCompletions = self._install_fake_openai()

        agentbudget.init(budget="$0.001")  # very small budget

        client = FakeCompletions()
        # gpt-4o: ~$0.0025 + $0.0005 per call with 100/50 tokens
        # Should exceed $0.001 budget
        with pytest.raises(agentbudget.BudgetExhausted):
            for _ in range(10):
                client.create(model="gpt-4o")


class TestConcurrency:
    def setup_method(self):
        agentbudget.teardown()

    def teardown_method(self):
        agentbudget.teardown()

    def test_thread_local_sessions_do_not_leak(self):
        modules, FakeCompletions, _ = _build_fake_openai_modules()
        start = threading.Barrier(2)
        reports = {}
        errors: list[Exception] = []

        def worker(session_id, tool_cost):
            try:
                agentbudget.init(budget="$5.00", session_id=session_id)
                client = FakeCompletions()
                start.wait(timeout=5)
                client.create(model="gpt-4o")
                agentbudget.track(cost=tool_cost, tool_name=f"tool_{session_id}")
                reports[session_id] = agentbudget.teardown()
            except Exception as exc:
                errors.append(exc)
                agentbudget.teardown()

        with mock.patch.dict(sys.modules, modules):
            threads = [
                threading.Thread(target=worker, args=("thread_a", 0.10)),
                threading.Thread(target=worker, args=("thread_b", 0.20)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert not errors
        assert agentbudget.get_session() is None

        first = reports["thread_a"]
        second = reports["thread_b"]

        assert first["session_id"] == "thread_a"
        assert second["session_id"] == "thread_b"
        assert first["breakdown"]["llm"]["calls"] == 1
        assert second["breakdown"]["llm"]["calls"] == 1
        assert set(first["breakdown"]["tools"]["by_tool"]) == {"tool_thread_a"}
        assert set(second["breakdown"]["tools"]["by_tool"]) == {"tool_thread_b"}
        assert first["breakdown"]["tools"]["by_tool"]["tool_thread_a"] == pytest.approx(0.10)
        assert second["breakdown"]["tools"]["by_tool"]["tool_thread_b"] == pytest.approx(0.20)

    def test_teardown_in_one_thread_keeps_other_thread_patched(self):
        modules, FakeCompletions, _ = _build_fake_openai_modules()
        ready = threading.Barrier(2)
        first_torn_down = threading.Event()
        reports = {}
        errors: list[Exception] = []

        def first_worker():
            try:
                agentbudget.init(budget="$5.00", session_id="first")
                ready.wait(timeout=5)
                reports["first"] = agentbudget.teardown()
            except Exception as exc:
                errors.append(exc)
                agentbudget.teardown()
            finally:
                first_torn_down.set()

        def second_worker():
            try:
                agentbudget.init(budget="$5.00", session_id="second")
                client = FakeCompletions()
                ready.wait(timeout=5)
                assert first_torn_down.wait(timeout=5)
                client.create(model="gpt-4o")
                reports["second_mid"] = agentbudget.report()
                reports["second"] = agentbudget.teardown()
            except Exception as exc:
                errors.append(exc)
                agentbudget.teardown()

        with mock.patch.dict(sys.modules, modules):
            threads = [
                threading.Thread(target=first_worker),
                threading.Thread(target=second_worker),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert not errors
        assert reports["first"]["session_id"] == "first"
        assert reports["second_mid"]["session_id"] == "second"
        assert reports["second_mid"]["breakdown"]["llm"]["calls"] == 1
        assert reports["second"]["breakdown"]["llm"]["calls"] == 1

    @pytest.mark.asyncio
    async def test_async_tasks_can_shadow_parent_session_independently(self):
        modules, _, FakeAsyncCompletions = _build_fake_openai_modules()

        with mock.patch.dict(sys.modules, modules):
            parent_session = agentbudget.init(budget="$5.00", session_id="parent")

            async def worker(session_id, tool_cost):
                try:
                    agentbudget.init(budget="$3.00", session_id=session_id)
                    client = FakeAsyncCompletions()
                    await client.create(model="gpt-4o", delay=0)
                    agentbudget.track(cost=tool_cost, tool_name=f"tool_{session_id}")
                    report = agentbudget.teardown()
                    restored = agentbudget.get_session()
                    return report, restored.session_id if restored else None
                finally:
                    current = agentbudget.get_session()
                    if current is not None and current.session_id == session_id:
                        agentbudget.teardown()

            child_reports = await asyncio.gather(
                worker("task_a", 0.10),
                worker("task_b", 0.20),
            )

            assert agentbudget.get_session() is parent_session
            assert agentbudget.report()["breakdown"]["llm"]["calls"] == 0
            parent_report = agentbudget.teardown()

        first_report, first_restored = child_reports[0]
        second_report, second_restored = child_reports[1]

        assert first_report["session_id"] == "task_a"
        assert second_report["session_id"] == "task_b"
        assert first_report["breakdown"]["llm"]["calls"] == 1
        assert second_report["breakdown"]["llm"]["calls"] == 1
        assert first_restored == "parent"
        assert second_restored == "parent"
        assert parent_report["session_id"] == "parent"
        assert parent_report["total_spent"] == 0.0
