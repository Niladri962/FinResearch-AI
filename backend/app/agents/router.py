"""Query understanding and intent routing.

A deterministic, explainable scorer handles the common cases. The LLM is asked
only when the rules are not confident and a provider is configured — most
queries never need a model call to be routed.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from sqlalchemy import select

from app.agents.state import CompanyRef, QueryUnderstanding
from app.documents.metadata import company_key
from app.financial.metrics import detect_metrics
from app.financial.periods import find_periods
from app.financial.ratios import RATIO_BY_KEY, TOPIC_RATIOS, detect_ratios
from app.llm.base import LLMClient
from app.models.database import SessionFactory
from app.models.db import Company
from app.models.enums import DocumentType, Intent
from app.models.schemas import QueryFilters
from app.utils.logging import get_logger

logger = get_logger(__name__)

_COMPARE = re.compile(r"\b(compare[ds]?|comparison|comparing|versus|vs\.?|against|relative to|better than|worse than|difference between|benchmark|stack up)\b", re.I)
_PEERS = re.compile(r"\b(peers?|competitors?|rivals?|industry|company a|company b|both companies|each other|other compan(y|ies))\b", re.I)
_YOY = re.compile(r"\b(year[- ]over[- ]year|year[- ]on[- ]year|yoy|y-o-y|quarter[- ]over[- ]quarter|qoq|previous year|prior year|last year|preceding year)\b", re.I)
_TREND_SPAN = re.compile(
    r"\b(over the (last|past|previous) (\w+ )?(years?|quarters?|periods?)|(last|past) (\d+|two|three|four|five|six|ten) years|"
    r"over time|over the years|historical(ly)?|cagr|trajectory|year[- ]wise|multi[- ]year|since (fy)?\s?\d{2,4}|trends?)\b", re.I)
_TREND_DIR = re.compile(r"\b(increasing|decreasing|declining|rising|falling|growing|shrinking|improving|deteriorating|worsening|accelerating|slowing)\b", re.I)
_TREND_HOW = re.compile(r"\bhow (has|have|did)\b.*\b(changed?|evolved?|moved?|progressed|trended|developed)\b", re.I)
_RISK = re.compile(r"\b(risks?|risk factors?|threats?|red flags?|concerns?|headwinds?|vulnerabilit\w+|challenges?|uncertaint\w+|downside|what could go wrong)\b", re.I)
_MGMT = re.compile(r"\b(management'?s?|guidance|outlook|expect(s|ed|ations?|ing)?|commentary|ceo|cfo|chairman|tone|sentiment|strateg(y|ic|ies)|plans?|priorit\w+|forward[- ]looking|next year|going forward)\b", re.I)
_SUMMARY = re.compile(r"\b(summari[sz]e|summari[sz]ation|summary|overview|key (takeaways|highlights|points)|highlights|recap|brief me|tl;?dr|gist|walk me through)\b", re.I)
_WHY = re.compile(r"^\s*(why\b|what (caused|drove|led to|explains?|is behind|are the drivers)|(what are|give|list|identify) (the )?(major |main |key |primary )?(reasons?|drivers?|factors?|causes?)|reasons? (for|behind))", re.I)
_CALC_VERB = re.compile(r"\b(calculate|compute|work out|what is|what's|what was|what are|how much|how many|tell me)\b", re.I)
_GENERAL = re.compile(
    r"^\s*(what does .+ (mean|measure|indicate|tell)|define\b|explain (what|the concept|how|the difference)|"
    r"what is (a|an|meant by)\b|what are (a|an)?\s*\w+ ratios?\s*\??$|how (is|are|do you|do i|to) .*\b(calculated?|computed?|interpret(ed)?|work)\b|"
    r"what('s| is) the (difference between|formula for|definition of|meaning of))", re.I)
_SPECIFIC = re.compile(r"\b(the company|company'?s|its|their|the firm|the business|our|this company|they|uploaded|document|report|filing)\b", re.I)
_FOLLOWUP = re.compile(r"^\s*(and|also|what about|how about|why|so|then|ok(ay)?[, ]|but)\b|\b(it|its|they|their|them|that|those|these|this|the same|above|previous answer)\b", re.I)

_DOC_TYPE_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(earnings call|conference call|concall|call transcript|transcript)\b", re.I), DocumentType.EARNINGS_CALL.value),
    (re.compile(r"\b(investor presentation|investor deck|results presentation)\b", re.I), DocumentType.INVESTOR_PRESENTATION.value),
    (re.compile(r"\b(annual report|10-k)\b", re.I), DocumentType.ANNUAL_REPORT.value),
    (re.compile(r"\b(quarterly report|quarterly results|10-q)\b", re.I), DocumentType.QUARTERLY_REPORT.value),
]

_CLASSIFY_PROMPT = """Classify the user's question for a financial research assistant into exactly one intent:
DOCUMENT_QA - factual or "why" question answered from the documents
FINANCIAL_CALCULATION - asks for a ratio or computed metric for a company
COMPANY_COMPARISON - compares two or more companies
PERIOD_COMPARISON - compares two or more reporting periods of one company
RISK_ANALYSIS - risks, threats, concerns
MANAGEMENT_ANALYSIS - management commentary, guidance, outlook, tone
SUMMARY - summarise a document or company performance
TREND_ANALYSIS - how something changed over several periods
GENERAL_FINANCE - generic finance concept, not about a specific company or document

Respond with JSON only: {"intent": "<INTENT>"}"""

_CONDENSE_PROMPT = (
    "Rewrite the follow-up question as a standalone question using the conversation for context. "
    "Keep company names, periods and metrics explicit. Do not answer it. Return only the rewritten question."
)


def score_intents(query: str, *, n_companies: int, n_years: int, ratios: list[str], topic_ratios: bool, metrics: list[str]) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    scores[Intent.DOCUMENT_QA.value] = 1.0
    compare = bool(_COMPARE.search(query))

    if n_companies >= 2:
        scores[Intent.COMPANY_COMPARISON.value] += 3.0 if compare else 1.5
    if compare and _PEERS.search(query):
        scores[Intent.COMPANY_COMPARISON.value] += 2.5
    if compare and n_companies < 2:
        scores[Intent.PERIOD_COMPARISON.value] += 3.0 if (n_years >= 2 or _YOY.search(query)) else 1.2
    elif _YOY.search(query):
        scores[Intent.PERIOD_COMPARISON.value] += 1.5

    if _TREND_SPAN.search(query):
        scores[Intent.TREND_ANALYSIS.value] += 4.0
    if _TREND_DIR.search(query):
        scores[Intent.TREND_ANALYSIS.value] += 1.5
    if _TREND_HOW.search(query):
        scores[Intent.TREND_ANALYSIS.value] += 2.0

    if _RISK.search(query):
        scores[Intent.RISK_ANALYSIS.value] += 3.0
    mgmt_hits = len({m.group(0).lower() for m in _MGMT.finditer(query)})
    if mgmt_hits:
        scores[Intent.MANAGEMENT_ANALYSIS.value] += 2.5 + min(mgmt_hits - 1, 2) * 0.5
    if _SUMMARY.search(query):
        scores[Intent.SUMMARY.value] += 3.5

    if ratios and not topic_ratios:
        scores[Intent.FINANCIAL_CALCULATION.value] += 3.0
        if _CALC_VERB.search(query):
            scores[Intent.FINANCIAL_CALCULATION.value] += 1.0
    elif ratios:
        scores[Intent.FINANCIAL_CALCULATION.value] += 1.5
    if re.search(r"\b(calculate|compute|ratios?)\b", query, re.I):
        scores[Intent.FINANCIAL_CALCULATION.value] += 1.5

    if _WHY.search(query):
        scores[Intent.DOCUMENT_QA.value] += 3.5

    if _GENERAL.search(query) and not _SPECIFIC.search(query) and n_companies == 0 and n_years == 0:
        scores[Intent.GENERAL_FINANCE.value] += 7.0
    return dict(scores)


def pick_intent(scores: dict[str, float]) -> tuple[Intent, float]:
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_intent, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = top / (top + second) if (top + second) else 0.0
    return Intent(top_intent), round(confidence, 3)


class QueryRouter:
    def __init__(self, session_factory: SessionFactory, llm: LLMClient) -> None:
        self._session_factory = session_factory
        self._llm = llm

    # ── Entity resolution ────────────────────────────────────────────────
    def _all_companies(self) -> list[Company]:
        with self._session_factory() as session:
            return list(session.execute(select(Company).order_by(Company.id)).scalars())

    @staticmethod
    def match_companies(query: str, companies: list[Company]) -> list[CompanyRef]:
        text = f" {re.sub(r'[^a-z0-9&]+', ' ', query.lower())} "
        first_words: dict[str, int] = defaultdict(int)
        for company in companies:
            first_words[company.key.split(" ")[0]] += 1
        found: list[tuple[int, CompanyRef]] = []
        for company in companies:
            names = {company.key, *(company_key(a) for a in company.aliases)}
            first = company.key.split(" ")[0]
            if len(first) >= 4 and first_words[first] == 1:
                names.add(first)  # "Aurora" is enough when only one company starts with it
            positions = [text.find(f" {name} ") for name in names if name]
            positions = [p for p in positions if p >= 0]
            if positions:
                found.append((min(positions), CompanyRef(id=company.id, name=company.name)))
        return [ref for _, ref in sorted(found, key=lambda item: item[0])]

    # ── LLM assists (optional) ───────────────────────────────────────────
    async def _llm_intent(self, query: str) -> Intent | None:
        try:
            response = await self._llm.complete(
                [{"role": "system", "content": _CLASSIFY_PROMPT}, {"role": "user", "content": query}],
                temperature=0.0, max_tokens=30, json_mode=True,
            )
            return Intent(json.loads(response.text)["intent"])
        except Exception as exc:  # routing must never fail the request
            logger.warning("LLM intent classification failed", extra={"error": type(exc).__name__})
            return None

    async def _condense(self, query: str, history: list[dict[str, str]]) -> str:
        transcript = "\n".join(f"{m['role'].upper()}: {m['content'][:500]}" for m in history[-4:])
        try:
            response = await self._llm.complete(
                [
                    {"role": "system", "content": _CONDENSE_PROMPT},
                    {"role": "user", "content": f"Conversation:\n{transcript}\n\nFollow-up question: {query}"},
                ],
                temperature=0.0, max_tokens=120,
            )
            rewritten = response.text.strip().strip('"')
            return rewritten if 5 <= len(rewritten) <= 600 else query
        except Exception as exc:
            logger.warning("Follow-up rewrite failed", extra={"error": type(exc).__name__})
            return query

    # ── Main entry point ─────────────────────────────────────────────────
    async def understand(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        filters: QueryFilters | None = None,
        context: dict | None = None,
    ) -> QueryUnderstanding:
        history, filters, context = history or [], filters or QueryFilters(), context or {}
        companies = self._all_companies()
        by_id = {c.id: c for c in companies}
        mentioned = self.match_companies(query, companies)

        is_followup = bool(history) and not mentioned and (bool(_FOLLOWUP.search(query)) or len(query.split()) <= 6)
        standalone = query
        if is_followup:
            if self._llm.available:
                standalone = await self._condense(query, history)
                mentioned = self.match_companies(standalone, companies)
            else:
                previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
                standalone = f"{query} (follow-up to: {previous[:200]})" if previous else query

        # Company scope: named in the question → UI filter → conversation context → the only company.
        resolved = mentioned
        if not resolved and filters.company_ids:
            resolved = [CompanyRef(id=i, name=by_id[i].name) for i in filters.company_ids if i in by_id]
        if not resolved and context.get("company_ids"):
            resolved = [CompanyRef(id=i, name=by_id[i].name) for i in context["company_ids"] if i in by_id]
        if not resolved and len(companies) == 1:
            resolved = [CompanyRef(id=companies[0].id, name=companies[0].name)]

        alias_ratios = detect_ratios(standalone, include_topics=False)
        ratios = alias_ratios or detect_ratios(standalone)
        # Strip ratio phrases first so "debt-to-equity" does not also register as "debt" and "equity".
        residual = standalone
        for topic in TOPIC_RATIOS:
            residual = re.sub(rf"\b{topic}\b", " ", residual, flags=re.I)
        for key in alias_ratios:
            for alias in RATIO_BY_KEY[key].aliases:
                residual = re.sub(re.escape(alias), " ", residual, flags=re.I)
        metrics = detect_metrics(residual)
        years = sorted({p.fiscal_year for p in find_periods(standalone)})
        doc_types = [dtype for pattern, dtype in _DOC_TYPE_HINTS if pattern.search(standalone)]

        n_companies = len(mentioned) if mentioned else (len(resolved) if len(filters.company_ids) >= 2 else min(len(resolved), 1))
        scores = score_intents(
            standalone, n_companies=n_companies, n_years=len(years),
            ratios=ratios, topic_ratios=bool(ratios) and not alias_ratios, metrics=metrics,
        )
        intent, confidence = pick_intent(scores)
        method = "rules"
        if confidence < 0.58 and self._llm.available:
            llm_intent = await self._llm_intent(standalone)
            if llm_intent is not None:
                intent, method = llm_intent, "llm"

        if intent == Intent.COMPANY_COMPARISON and len(resolved) < 2 and len(companies) == 2:
            resolved = [CompanyRef(id=c.id, name=c.name) for c in companies]
        if intent == Intent.GENERAL_FINANCE:
            resolved = mentioned

        return QueryUnderstanding(
            intent=intent, confidence=confidence, method=method, query=standalone,
            companies=resolved, fiscal_years=years, ratios=ratios, metrics=metrics,
            document_types=doc_types, is_followup=is_followup,
            scores={k: round(v, 2) for k, v in scores.items()},
        )
