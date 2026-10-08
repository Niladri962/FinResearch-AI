"""Financial agent: ratios, metric look-ups and trends — all computed in Python."""
from __future__ import annotations

import asyncio
from typing import Any

from app.agents.state import AgentDeps, AgentState, RunContext, assign_ids
from app.financial.ratios import RATIOS
from app.models.enums import Intent
from app.models.schemas import Calculation, ChartSpec, DataTable
from app.services.financials import KPI_KEYS

_DEFAULT_TREND_KEYS = ["revenue", "net_income", "operating_margin", "net_margin"]
_SUMMARY_KEYS = [*KPI_KEYS, "operating_margin", "net_margin", "revenue_growth"]
_NUMERIC_INTENTS = {Intent.FINANCIAL_CALCULATION, Intent.TREND_ANALYSIS}
MAX_COMPANIES = 3
MAX_KEYS = 8


class FinancialAgent:
    name = "financial"

    def __init__(self, deps: AgentDeps) -> None:
        self._deps = deps

    def _compute(self, state: AgentState) -> tuple[list[Calculation], list[DataTable], list[ChartSpec], list[str]]:
        understanding = state["understanding"]
        fin = self._deps.financials
        intent = understanding.intent
        requested = list(dict.fromkeys([*understanding.ratios, *understanding.metrics]))
        calcs: list[Calculation] = []
        tables: list[DataTable] = []
        charts: list[ChartSpec] = []
        notes: list[str] = []

        if not understanding.companies:
            # Only worth telling the user when figures are the point of the question.
            if intent in _NUMERIC_INTENTS:
                notes.append("No company could be identified for the calculation. Select a company or name it in the question.")
            return calcs, tables, charts, notes

        for company in understanding.companies[:MAX_COMPANIES]:
            periods = fin.periods(company.id)
            if not periods:
                notes.append(
                    f"No structured financial statements were extracted for {company.name}, so ratios cannot be "
                    "calculated. Any figures in the answer come from retrieved text only."
                )
                continue

            if intent == Intent.TREND_ANALYSIS:
                keys = (requested or _DEFAULT_TREND_KEYS)[:4]
                t, c, k = fin.trend_artifacts(company.name, periods, keys)
                tables.extend(t)
                charts.extend(c)
                calcs.extend(k)
                if len(periods) < 2:
                    notes.append(f"Only one fiscal year of data is available for {company.name}; a trend needs at least two.")
                elif not t:
                    notes.append(f"The requested metrics were not found in {company.name}'s extracted statements.")
                continue

            if intent == Intent.SUMMARY:
                keys = _SUMMARY_KEYS
            elif requested:
                keys = requested[:MAX_KEYS]
            elif intent == Intent.FINANCIAL_CALCULATION:
                keys = [r.key for r in RATIOS]
            else:
                continue

            wanted_years = [y for y in understanding.fiscal_years if any(p.fiscal_year == y for p in periods)]
            missing_years = [y for y in understanding.fiscal_years if y not in wanted_years]
            if missing_years:
                notes.append(
                    f"No structured data for {company.name} in " + ", ".join(f"FY{y}" for y in missing_years)
                    + f". Available: {', '.join(p.label for p in periods)}."
                )
            targets = wanted_years[-3:] or ([periods[-1].fiscal_year] if not understanding.fiscal_years else [])
            for year in targets:
                period, previous = fin.pick(periods, year)
                if period is None:
                    continue
                calcs.extend(fin.calculate(company.name, key, period, previous) for key in keys)
        return calcs, tables, charts, notes

    async def run(self, state: AgentState, run: RunContext) -> dict[str, Any]:
        run.status("financial", "Calculating financial metrics…")
        with run.trace.span("financial_calculations"):
            calcs, tables, charts, notes = await asyncio.to_thread(self._compute, state)
        update = assign_ids(state, calcs, tables, charts)
        update["notes"] = [*state.get("notes", []), *notes]
        run.trace.set(calculations=len(update["calculations"]))
        return update
