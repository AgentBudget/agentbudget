"""Tests for framework integrations."""

from uuid import uuid4

import pytest

from agentbudget import AgentBudget, BudgetExhausted
from agentbudget.integrations.crewai import CrewAIBudgetMiddleware
from agentbudget.integrations import langchain as langchain_module


def test_langchain_import_without_langchain():
    """Should raise ImportError when langchain-core is not installed."""
    from agentbudget.integrations.langchain import LangChainBudgetCallback

    with pytest.raises(ImportError, match="langchain-core"):
        LangChainBudgetCallback(budget="$5.00")


class _FakeLLMResult:
    def __init__(self, *, llm_output=None, generations=None):
        self.llm_output = llm_output or {}
        self.generations = generations or []


class _FakeMessage:
    def __init__(self, usage_metadata=None, response_metadata=None, additional_kwargs=None):
        self.usage_metadata = usage_metadata or {}
        self.response_metadata = response_metadata or {}
        self.additional_kwargs = additional_kwargs or {}


class _FakeGeneration:
    def __init__(self, message):
        self.message = message


class TestLangChainBudgetCallback:
    def test_tracks_llm_cost_from_llm_output(self, monkeypatch):
        monkeypatch.setattr(langchain_module, "_HAS_LANGCHAIN", True)

        callback = langchain_module.LangChainBudgetCallback(budget="$5.00")
        run_id = uuid4()

        response = _FakeLLMResult(
            llm_output={
                "model_name": "gpt-4o",
                "token_usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                },
            }
        )

        callback.on_llm_end(response, run_id=run_id)
        report = callback.close()

        assert report["total_spent"] == pytest.approx(0.0075, rel=1e-4)

    def test_tracks_usage_metadata_when_llm_output_is_missing(self, monkeypatch):
        monkeypatch.setattr(langchain_module, "_HAS_LANGCHAIN", True)

        callback = langchain_module.LangChainBudgetCallback(budget="$5.00")
        run_id = uuid4()

        callback.on_chat_model_start(
            {"name": "ChatOpenAI"},
            messages=[],
            run_id=run_id,
            invocation_params={"model": "gpt-4o"},
        )
        response = _FakeLLMResult(
            generations=[
                [
                    _FakeGeneration(
                        _FakeMessage(
                            usage_metadata={
                                "input_tokens": 1000,
                                "output_tokens": 500,
                            }
                        )
                    )
                ]
            ]
        )

        callback.on_llm_end(response, run_id=run_id)
        report = callback.close()

        assert report["total_spent"] == pytest.approx(0.0075, rel=1e-4)

    def test_tracks_tool_costs_from_metadata(self, monkeypatch):
        monkeypatch.setattr(langchain_module, "_HAS_LANGCHAIN", True)

        callback = langchain_module.LangChainBudgetCallback(budget="$5.00")
        run_id = uuid4()

        callback.on_tool_start(
            {"name": "search"},
            "weather in sf",
            run_id=run_id,
            metadata={"agentbudget_cost": 0.25, "provider": "serpapi"},
        )
        callback.on_tool_end("sunny", run_id=run_id)
        report = callback.close()

        assert report["total_spent"] == pytest.approx(0.25)
        assert report["events"][0]["tool_name"] == "search"
        assert report["events"][0]["metadata"]["provider"] == "serpapi"

    def test_tracks_tool_costs_from_mapping_and_extractor(self, monkeypatch):
        monkeypatch.setattr(langchain_module, "_HAS_LANGCHAIN", True)

        extractor_calls = []
        callback = langchain_module.LangChainBudgetCallback(
            budget="$5.00",
            tool_costs={"search": 0.10},
            tool_cost_extractor=lambda tool_name, output, metadata: (
                extractor_calls.append((tool_name, output, metadata)) or 0.20
            ),
        )

        mapped_run_id = uuid4()
        callback.on_tool_start({"name": "search"}, "prompt", run_id=mapped_run_id)
        callback.on_tool_end("mapped", run_id=mapped_run_id)

        extracted_run_id = uuid4()
        callback.on_tool_start(
            {"name": "fetch"},
            "prompt",
            run_id=extracted_run_id,
            metadata={"source": "api"},
        )
        callback.on_tool_end("extracted", run_id=extracted_run_id)
        report = callback.close()

        assert report["total_spent"] == pytest.approx(0.30)
        assert extractor_calls == [("fetch", "extracted", {"source": "api"})]

    def test_does_not_own_user_supplied_session(self, monkeypatch):
        monkeypatch.setattr(langchain_module, "_HAS_LANGCHAIN", True)

        budget = AgentBudget(max_spend="$5.00")
        session = budget.session()
        callback = langchain_module.LangChainBudgetCallback(
            budget="$5.00",
            session=session,
        )

        assert session._start_time is None
        report = callback.close()

        assert session._end_time is None
        assert report["duration_seconds"] is None

    def test_context_manager_closes_owned_session(self, monkeypatch):
        monkeypatch.setattr(langchain_module, "_HAS_LANGCHAIN", True)

        run_id = uuid4()
        with langchain_module.LangChainBudgetCallback(budget="$5.00") as callback:
            callback.on_tool_start(
                {"name": "search"},
                "query",
                run_id=run_id,
                metadata={"agentbudget_cost": 0.05},
            )
            callback.on_tool_end("done", run_id=run_id)

        report = callback.get_report()
        assert report["duration_seconds"] is not None
        assert report["total_spent"] == pytest.approx(0.05)


class TestCrewAIMiddleware:
    def test_basic_creation(self):
        mw = CrewAIBudgetMiddleware(budget="$5.00")
        assert mw.session.remaining == 5.0

    def test_context_manager(self):
        with CrewAIBudgetMiddleware(budget="$5.00") as mw:
            mw.track("result", cost=0.50, tool_name="search")
            assert mw.session.spent == 0.50

    def test_budget_enforcement(self):
        with pytest.raises(BudgetExhausted):
            with CrewAIBudgetMiddleware(budget="$0.10") as mw:
                mw.track("a", cost=0.06)
                mw.track("b", cost=0.06)

    def test_report(self):
        with CrewAIBudgetMiddleware(budget="$5.00", session_id="sess_crew") as mw:
            mw.track("x", cost=1.0, tool_name="tool")

        report = mw.get_report()
        assert report["session_id"] == "sess_crew"
        assert report["total_spent"] == 1.0

    def test_callbacks(self):
        warnings = []
        with CrewAIBudgetMiddleware(
            budget="$1.00",
            on_soft_limit=lambda r: warnings.append(r),
        ) as mw:
            mw.track("x", cost=0.95, tool_name="tool")

        assert len(warnings) == 1
