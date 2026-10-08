"""Summary agent: turns evidence and computed figures into the final answer.

With an LLM configured the answer is generated and streamed. Without one — or
if the provider fails mid-request — the agent returns an *extractive* answer:
the computed metrics plus the most relevant passages quoted verbatim. Either
way nothing is stated that is not in the evidence or the calculations.
"""
from __future__ import annotations

from typing import Any

from app.agents.prompts import build_general_messages, build_messages
from app.agents.state import AgentDeps, AgentState, RunContext
from app.documents.table_extractor import table_to_markdown
from app.financial.ratios import RATIO_BY_KEY
from app.guardrails.output_guard import INSUFFICIENT_EVIDENCE
from app.llm.base import LLMUsage
from app.models.enums import ChunkType, Intent
from app.models.schemas import Calculation, DataTable, RetrievedChunk
from app.utils.errors import LLMError
from app.utils.logging import get_logger
from app.utils.text import split_sentences, tokenize, truncate

logger = get_logger(__name__)

EXTRACTIVE_BANNER = (
    "_Extractive mode: no language model is configured, so this answer lists the computed metrics and the most "
    "relevant passages from your documents verbatim._"
)
LLM_FAILED_BANNER = (
    "_The language model was unavailable ({reason}), so this answer lists the computed metrics and the most "
    "relevant passages from your documents verbatim._"
)
_TABLE_TYPES = {ChunkType.FINANCIAL_STATEMENT.value, ChunkType.TABLE.value}


def _best_sentences(query: str, chunk: RetrievedChunk, limit: int = 2) -> str:
    if chunk.chunk_type in _TABLE_TYPES:
        first_line = chunk.text.splitlines()[0] if chunk.text else ""
        return f"Table: {truncate(first_line, 140)}"
    terms = set(tokenize(query))
    paragraphs = [p for p in chunk.text.split("\n\n") if p.strip()]
    # Chunks open with their section heading; it is context, not a quotable sentence.
    if len(paragraphs) > 1 and len(paragraphs[0].split()) <= 14 and paragraphs[0].strip()[-1:] not in ".!?":
        paragraphs = paragraphs[1:]
    sentences = split_sentences(" ".join(paragraphs).replace("\n", " "))
    if not sentences:
        return truncate(chunk.text, 300)
    scored = sorted(
        enumerate(sentences),
        key=lambda item: (-len(terms & set(tokenize(item[1]))), item[0]),
    )[:limit]
    return truncate(" ".join(s for _, s in sorted(scored)), 420)


def compose_extractive(
    query: str, evidence: list[RetrievedChunk], calculations: list[Calculation],
    tables: list[DataTable], notes: list[str], banner: str = EXTRACTIVE_BANNER,
) -> str:
    parts: list[str] = [banner]
    metrics = [c for c in calculations if c.kind != "flag"]
    flags = [c for c in calculations if c.kind == "flag"]
    if metrics:
        rows = [["Metric", "Company", "Period", "Value", "Formula", "Ref"]]
        rows += [[c.name, c.company, c.period, c.display, c.formula, f"[{c.id}]"] for c in metrics]
        parts.append("### Calculated metrics\n" + table_to_markdown(rows))
        unavailable = [c for c in metrics if c.value is None and c.note]
        if unavailable:
            parts.append("\n".join(f"- {c.name}: {c.note}" for c in unavailable))
    if flags:
        parts.append("### Quantitative risk signals\n" + "\n".join(
            f"- **{c.name}** ({c.severity}): {c.display} [{c.id}]" for c in flags
        ))
    for table in tables:
        block = f"### {table.title} [{table.id}]\n" + table_to_markdown([table.columns, *table.rows])
        if table.note:
            block += f"\n\n_{table.note}_"
        parts.append(block)
    if evidence:
        lines = [
            f"- {_best_sentences(query, chunk)} [S{index}]"
            for index, chunk in enumerate(evidence[:6], start=1)
        ]
        parts.append("### Most relevant evidence\n" + "\n".join(lines))
    if notes:
        parts.append("### Data notes\n" + "\n".join(f"- {n}" for n in notes))
    return "\n\n".join(parts)


def glossary(ratio_keys: list[str]) -> str:
    return "\n".join(
        f"- {RATIO_BY_KEY[k].name}: {RATIO_BY_KEY[k].description} Formula: {RATIO_BY_KEY[k].formula}."
        for k in ratio_keys if k in RATIO_BY_KEY
    )


class SummaryAgent:
    name = "summary"

    def __init__(self, deps: AgentDeps) -> None:
        self._deps = deps

    async def _generate(self, messages: list[dict[str, str]], run: RunContext, prefix: str) -> str:
        """Stream a completion to the client and return the full text."""
        usage = LLMUsage()
        pieces: list[str] = []
        if prefix:
            run.emit("token", {"text": prefix})
        with run.trace.span("llm"):
            async for delta in self._deps.llm.stream(messages, usage=usage):
                pieces.append(delta)
                run.emit("token", {"text": delta})
        run.trace.add_usage(usage.as_dict())
        run.trace.set(llm_model=self._deps.llm.model, llm_provider=self._deps.llm.provider)
        return prefix + "".join(pieces).strip()

    async def run(self, state: AgentState, run: RunContext) -> dict[str, Any]:
        understanding = state["understanding"]
        evidence = state.get("evidence", [])
        calculations = state.get("calculations", [])
        tables = state.get("tables", [])
        notes = state.get("notes", [])
        notice = state.get("guard_notice", "")
        prefix = f"{notice}\n\n" if notice else ""
        llm = self._deps.llm

        # ── General finance: no document grounding required ─────────────
        if understanding.intent == Intent.GENERAL_FINANCE:
            reference = glossary(understanding.ratios)
            if llm.available:
                run.status("generation", "Writing the explanation…")
                try:
                    answer = await self._generate(build_general_messages(understanding.query, reference), run, prefix)
                    return {"answer": answer, "mode": "general_knowledge"}
                except LLMError as exc:
                    logger.warning("LLM failed for general question", extra={"code": exc.code})
            body = (
                "_General finance explanation — not based on your uploaded documents._\n\n" + reference
                if reference else
                "This looks like a general finance question. A language model is needed to answer it, and none is "
                "configured. Set `LLM_API_KEY` to enable general explanations, or ask about your uploaded documents."
            )
            answer = prefix + body
            run.emit("token", {"text": answer})
            return {"answer": answer, "mode": "general_knowledge"}

        # ── Nothing to ground an answer on ──────────────────────────────
        has_numbers = any(c.value is not None or c.kind == "flag" for c in calculations)
        if not evidence and not has_numbers and not tables:
            answer = prefix + INSUFFICIENT_EVIDENCE
            if notes:
                answer += "\n\n" + "\n".join(f"- {n}" for n in notes)
            run.emit("token", {"text": answer})
            return {"answer": answer, "mode": "insufficient_evidence"}

        # ── Generative answer ───────────────────────────────────────────
        if llm.available:
            run.status("generation", "Writing the answer…")
            messages = build_messages(
                question=understanding.query, intent=understanding.intent, evidence=evidence,
                calculations=calculations, tables=tables, history=state.get("history", []), notes=notes,
            )
            try:
                answer = await self._generate(messages, run, prefix)
                if answer.strip():
                    mode = "insufficient_evidence" if INSUFFICIENT_EVIDENCE[:60] in answer and len(answer) < 400 else "generative"
                    return {"answer": answer, "mode": mode}
                banner = LLM_FAILED_BANNER.format(reason="empty response")
            except LLMError as exc:
                logger.warning("LLM generation failed; falling back to extractive answer", extra={"code": exc.code})
                run.trace.set(llm_error=exc.code)
                banner = LLM_FAILED_BANNER.format(reason=exc.message.rstrip("."))
            # Tell the client to discard any partial stream before the fallback text arrives.
            run.emit("reset", {})
        else:
            banner = EXTRACTIVE_BANNER

        # ── Extractive answer ───────────────────────────────────────────
        answer = prefix + compose_extractive(understanding.query, evidence, calculations, tables, notes, banner)
        run.emit("token", {"text": answer})
        return {"answer": answer, "mode": "extractive"}
