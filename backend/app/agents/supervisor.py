"""Supervisor: the LangGraph that routes a query through the specialist agents.

    guard → understand → supervisor ⇄ {research | financial | comparison | risk}
                              └────→ summary → verify → END

The supervisor node decides the next agent from a plan derived from the query
intent. Agents return state updates; only ``summary`` talks to the LLM and only
the financial/comparison/risk agents produce numbers.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from langgraph.graph import END, START, StateGraph

from app.agents.comparison_agent import ComparisonAgent
from app.agents.financial_agent import FinancialAgent
from app.agents.research_agent import ResearchAgent
from app.agents.risk_agent import RiskAgent
from app.agents.router import QueryRouter
from app.agents.state import AgentDeps, AgentState, QueryUnderstanding, RunContext
from app.agents.summarization_agent import SummaryAgent
from app.guardrails.citation_validator import build_citations, validate_answer
from app.guardrails.input_guard import check_input
from app.guardrails.output_guard import check_output
from app.models.enums import Intent
from app.models.schemas import QueryFilters, ValidationReport
from app.utils.logging import get_logger
from app.utils.observability import Trace

logger = get_logger(__name__)

_BASE_PLANS: dict[Intent, list[str]] = {
    Intent.DOCUMENT_QA: ["research"],
    Intent.FINANCIAL_CALCULATION: ["financial", "research"],
    Intent.COMPANY_COMPARISON: ["comparison", "research"],
    Intent.PERIOD_COMPARISON: ["comparison", "research"],
    Intent.RISK_ANALYSIS: ["risk", "research"],
    Intent.MANAGEMENT_ANALYSIS: ["research"],
    Intent.SUMMARY: ["financial", "research"],
    Intent.TREND_ANALYSIS: ["financial", "research"],
    Intent.GENERAL_FINANCE: [],
}
_STAGE_LABELS = {
    "research": "Research agent: retrieving evidence",
    "financial": "Financial agent: computing metrics",
    "comparison": "Comparison agent: building comparison",
    "risk": "Risk agent: screening for risk signals",
    "summary": "Summary agent: composing the answer",
}


def build_plan(understanding: QueryUnderstanding) -> list[str]:
    plan = list(_BASE_PLANS[understanding.intent])
    # A document question that names a metric also gets the computed figures for it.
    if understanding.needs_numbers and plan and not {"financial", "comparison"} & set(plan):
        plan.insert(0, "financial")
    return plan


def suggest_followups(understanding: QueryUnderstanding) -> list[str]:
    company = understanding.companies[0].name if understanding.companies else "the company"
    by_intent: dict[Intent, list[str]] = {
        Intent.DOCUMENT_QA: [
            f"What are the biggest risks for {company}?",
            f"How has {company}'s profitability changed over time?",
            f"What does management expect for next year?",
        ],
        Intent.FINANCIAL_CALCULATION: [
            f"How has this ratio changed over time for {company}?",
            f"What explains the change, according to management?",
            f"Show the full set of liquidity and solvency ratios for {company}.",
        ],
        Intent.COMPANY_COMPARISON: [
            "Which company has the stronger balance sheet, based on the ratios?",
            "How do the two companies' risk factors differ?",
            "Compare management's outlook for both companies.",
        ],
        Intent.PERIOD_COMPARISON: [
            f"Why did {company}'s margins change between these periods?",
            f"How did {company}'s debt and cash flow change?",
            f"What are {company}'s capital expenditure plans?",
        ],
        Intent.RISK_ANALYSIS: [
            f"Is {company}'s debt increasing?",
            f"What is {company}'s interest coverage?",
            "How does management plan to mitigate these risks?",
        ],
        Intent.MANAGEMENT_ANALYSIS: [
            f"What guidance did management give on margins?",
            f"What are {company}'s capital expenditure plans?",
            f"What risks did management highlight?",
        ],
        Intent.SUMMARY: [
            f"What are {company}'s key financial ratios?",
            f"What were the main reasons for the change in net profit?",
            f"What are the biggest risks for {company}?",
        ],
        Intent.TREND_ANALYSIS: [
            f"What drove the trend, according to management?",
            f"How have {company}'s debt and cash flow trended?",
            f"Compare the latest two fiscal years for {company}.",
        ],
        Intent.GENERAL_FINANCE: [
            "Calculate this ratio for one of my uploaded companies.",
            "What is a healthy range for this metric?",
        ],
    }
    return by_intent.get(understanding.intent, [])[:3]


class Supervisor:
    def __init__(self, deps: AgentDeps) -> None:
        self._deps = deps
        self._router = QueryRouter(deps.session_factory, deps.llm)
        self._agents = {
            "research": ResearchAgent(deps),
            "financial": FinancialAgent(deps),
            "comparison": ComparisonAgent(deps),
            "risk": RiskAgent(deps),
        }
        self._summary = SummaryAgent(deps)

    @property
    def router(self) -> QueryRouter:
        return self._router

    # ── Graph construction ───────────────────────────────────────────────
    def _compile(self, run: RunContext):  # noqa: ANN202
        deps = self._deps

        async def guard(state: AgentState) -> dict[str, Any]:
            result = check_input(state["query"], max_chars=deps.settings.max_query_chars)
            run.trace.set(guard_action=result.action, guard_categories=result.categories)
            if result.blocked:
                run.emit("token", {"text": result.message})
                return {
                    "blocked": True, "answer": result.message, "mode": "blocked",
                    "guard_categories": result.categories, "understanding": QueryUnderstanding(query=state["query"]),
                }
            return {"blocked": False, "guard_notice": result.message, "guard_categories": result.categories}

        async def understand(state: AgentState) -> dict[str, Any]:
            run.status("understanding", "Understanding the question…")
            with run.trace.span("query_understanding"):
                understanding = await self._router.understand(
                    state["query"], state.get("history", []), state.get("filters"), state.get("context", {})
                )
            plan = build_plan(understanding)
            run.trace.set(intent=understanding.intent.value, intent_confidence=understanding.confidence,
                          intent_method=understanding.method, plan=plan)
            run.emit("meta", {
                "intent": understanding.intent.value,
                "confidence": understanding.confidence,
                "method": understanding.method,
                "companies": [c.model_dump() for c in understanding.companies],
                "fiscal_years": understanding.fiscal_years,
                "rewritten_query": understanding.query if understanding.query != state["query"] else None,
                "plan": [*plan, "summary"],
            })
            return {"understanding": understanding, "plan": plan, "cursor": 0}

        async def supervisor(state: AgentState) -> dict[str, Any]:
            plan, cursor = state.get("plan", []), state.get("cursor", 0)
            if cursor < len(plan):
                return {"next": plan[cursor], "cursor": cursor + 1}
            # Everything gathered: show the artifacts while the answer is being written.
            run.emit("artifacts", {
                "calculations": [c.model_dump() for c in state.get("calculations", [])],
                "tables": [t.model_dump() for t in state.get("tables", [])],
                "charts": [c.model_dump() for c in state.get("charts", [])],
            })
            run.emit("sources", {"sources": [c.model_dump() for c in build_citations(state.get("evidence", []))]})
            return {"next": "summary"}

        def agent_node(name: str):  # noqa: ANN202
            async def node(state: AgentState) -> dict[str, Any]:
                run.status(name, _STAGE_LABELS[name])
                return await self._agents[name].run(state, run)
            return node

        async def summary(state: AgentState) -> dict[str, Any]:
            run.status("summary", _STAGE_LABELS["summary"])
            return await self._summary.run(state, run)

        async def verify(state: AgentState) -> dict[str, Any]:
            run.status("verification", "Verifying citations and figures…")
            evidence = state.get("evidence", [])
            sources = build_citations(evidence)
            mode = state.get("mode", "generative")
            with run.trace.span("verification"):
                if mode in ("generative", "extractive"):
                    answer, report = validate_answer(
                        state.get("answer", ""), evidence=evidence,
                        calculations=state.get("calculations", []), tables=state.get("tables", []),
                        charts=state.get("charts", []), query=state["query"],
                    )
                else:
                    answer, report = state.get("answer", ""), ValidationReport()
                # The guard's own opening notice talks about guarantees; check only what follows it.
                notice = state.get("guard_notice", "")
                if notice and answer.startswith(notice):
                    guarded = check_output(answer[len(notice):])
                    guarded.text = notice + guarded.text
                else:
                    guarded = check_output(answer)
            cited = set(report.cited_ids)
            run.trace.set(
                validation_status=report.status, grounding_score=report.grounding_score,
                invalid_citations=len(report.invalid_citations), unsupported_numbers=len(report.unsupported_numbers),
                output_flags=guarded.flags, mode=mode,
            )
            return {
                "answer": guarded.text,
                "validation": report,
                "sources": sources,
                "citations": [c for c in sources if c.id in cited],
                "followups": suggest_followups(state["understanding"]),
            }

        graph = StateGraph(AgentState)
        graph.add_node("guard", guard)
        graph.add_node("understand", understand)
        graph.add_node("supervisor", supervisor)
        for name in self._agents:
            graph.add_node(name, agent_node(name))
        graph.add_node("summary", summary)
        graph.add_node("verify", verify)

        graph.add_edge(START, "guard")
        graph.add_conditional_edges("guard", lambda s: "blocked" if s.get("blocked") else "ok",
                                    {"blocked": END, "ok": "understand"})
        graph.add_edge("understand", "supervisor")
        graph.add_conditional_edges("supervisor", lambda s: s["next"],
                                    {**{name: name for name in self._agents}, "summary": "summary"})
        for name in self._agents:
            graph.add_edge(name, "supervisor")
        graph.add_edge("summary", "verify")
        graph.add_edge("verify", END)
        return graph.compile()

    # ── Execution ────────────────────────────────────────────────────────
    async def stream(
        self,
        *,
        query: str,
        trace: Trace,
        history: list[dict[str, str]] | None = None,
        filters: QueryFilters | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Run the graph, yielding ``(event, data)`` as work progresses.

        The last event is always ``("result", {"state": AgentState})`` or ``("error", {...})``.
        """
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        run = RunContext(trace=trace, emit=lambda event, data: queue.put_nowait((event, data)))
        graph = self._compile(run)
        initial: AgentState = {
            "query": query, "history": history or [], "filters": filters or QueryFilters(),
            "context": context or {}, "evidence": [], "calculations": [], "tables": [], "charts": [], "notes": [],
        }

        async def execute() -> None:
            try:
                state = await graph.ainvoke(initial, {"recursion_limit": 30})
                queue.put_nowait(("result", {"state": state}))
            except Exception as exc:  # surfaced to the caller as an event, never swallowed
                queue.put_nowait(("error", {"exception": exc}))

        task = asyncio.create_task(execute())
        try:
            while True:
                event, data = await queue.get()
                yield event, data
                if event in ("result", "error"):
                    break
        finally:
            if not task.done():
                task.cancel()  # client went away: stop retrieval/generation

    async def run(self, **kwargs: Any) -> AgentState:
        """Non-streaming convenience wrapper around :meth:`stream`."""
        async for event, data in self.stream(**kwargs):
            if event == "result":
                return data["state"]
            if event == "error":
                raise data["exception"]
        raise RuntimeError("Agent graph ended without a result")  # pragma: no cover
