"""Comparison agent: company-vs-company and period-vs-period tables."""
from __future__ import annotations

import asyncio
from typing import Any

from app.agents.state import AgentDeps, AgentState, RunContext, assign_ids
from app.models.enums import Intent
from app.services.comparison import ComparisonResult


class ComparisonAgent:
    name = "comparison"

    def __init__(self, deps: AgentDeps) -> None:
        self._deps = deps

    def _compare(self, state: AgentState) -> tuple[ComparisonResult | None, list[str]]:
        understanding = state["understanding"]
        companies = [(c.id, c.name) for c in understanding.companies]
        years = understanding.fiscal_years

        if understanding.intent == Intent.COMPANY_COMPARISON:
            if len(companies) < 2:
                return None, [
                    "A company comparison needs two companies with uploaded documents. Name both companies or "
                    "select them in the filter."
                ]
            result = self._deps.comparison.compare_companies(companies[:4], years[-1] if len(years) == 1 else None)
        else:
            if not companies:
                return None, ["No company could be identified for the period comparison."]
            company_id, name = companies[0]
            result = self._deps.comparison.compare_periods(company_id, name, years if len(years) >= 2 else None)
        return result, list(result.notes)

    async def run(self, state: AgentState, run: RunContext) -> dict[str, Any]:
        run.status("comparison", "Building the comparison…")
        with run.trace.span("comparison"):
            result, notes = await asyncio.to_thread(self._compare, state)
        update: dict[str, Any] = {"notes": [*state.get("notes", []), *notes]}
        if result is not None and result.has_data:
            update.update(assign_ids(state, [], [result.table], result.charts))
        return update
