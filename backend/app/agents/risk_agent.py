"""Risk agent: quantitative risk signals computed from reported figures.

Qualitative risk disclosures are gathered by the research agent, which the
supervisor schedules right after this one for risk questions.
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.agents.state import AgentDeps, AgentState, RunContext, assign_ids
from app.models.schemas import Calculation

MAX_COMPANIES = 3


class RiskAgent:
    name = "risk"

    def __init__(self, deps: AgentDeps) -> None:
        self._deps = deps

    def _signals(self, state: AgentState) -> list[Calculation]:
        flags: list[Calculation] = []
        for company in state["understanding"].companies[:MAX_COMPANIES]:
            periods = self._deps.financials.periods(company.id)
            flags.extend(self._deps.financials.risk_flags(company.name, periods))
        return flags

    async def run(self, state: AgentState, run: RunContext) -> dict[str, Any]:
        run.status("risk", "Screening financial statements for risk signals…")
        with run.trace.span("risk_signals"):
            flags = await asyncio.to_thread(self._signals, state)
        run.trace.set(risk_signals=len(flags))
        return assign_ids(state, flags, [], [])
