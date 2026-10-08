"""End-to-end RAG evaluation over a labelled dataset.

Dataset rows (JSON Lines):
    {"question", "expected_answer", "relevant_document", "relevant_page", "key_facts": [...]}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.evaluation.answer_eval import score_answer
from app.evaluation.retrieval_eval import mean, score_retrieval
from app.models.schemas import RetrievedChunk
from app.services.container import AppContainer


@dataclass
class EvalRow:
    question: str
    expected_answer: str
    relevant_document: str
    relevant_page: int | None = None
    key_facts: list[str] = field(default_factory=list)


def load_dataset(path: Path) -> list[EvalRow]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                rows.append(EvalRow(
                    question=data["question"], expected_answer=data["expected_answer"],
                    relevant_document=data["relevant_document"], relevant_page=data.get("relevant_page"),
                    key_facts=list(data.get("key_facts", [])),
                ))
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{number}: invalid evaluation row ({exc})") from exc
    return rows


def is_relevant(chunk: RetrievedChunk, row: EvalRow) -> bool:
    if chunk.filename != row.relevant_document:
        return False
    return row.relevant_page is None or chunk.page_start <= row.relevant_page <= chunk.page_end


async def run_evaluation(container: AppContainer, rows: list[EvalRow], *, k: int = 5) -> dict[str, Any]:
    """Evaluate retrieval (top ``k``) and the generated answer for every row."""
    details: list[dict[str, Any]] = []
    for row in rows:
        retrieval = container.retriever.retrieve(row.question, top_n=k)
        relevance = [is_relevant(c, row) for c in retrieval.chunks]
        # One labelled passage per question, so recall@k is 1 when it is found.
        r_scores = score_retrieval(
            row.question, relevance, [c.text for c in retrieval.chunks], total_relevant=1, k=k
        )

        state = await container.chat.ask(row.question)
        answer = state.get("answer", "")
        evidence = state.get("evidence", [])
        contexts = [c.text for c in evidence]
        # Computed figures are legitimate grounding too.
        contexts += [f"{c.name} {c.display} {' '.join(i.display for i in c.inputs)}" for c in state.get("calculations", [])]
        contexts += [" ".join(" ".join(r) for r in t.rows) for t in state.get("tables", [])]
        a_scores = score_answer(
            question=row.question, answer=answer, contexts=contexts,
            citations=state.get("citations", []) or state.get("sources", []),
            key_facts=row.key_facts, relevant_document=row.relevant_document, relevant_page=row.relevant_page,
        )
        validation = state.get("validation")
        details.append({
            "question": row.question,
            "intent": state["understanding"].intent.value,
            "mode": state.get("mode"),
            "retrieval": r_scores.as_dict(),
            "generation": a_scores.as_dict(),
            "grounding_score": validation.grounding_score if validation else None,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": row.expected_answer,
        })

    def avg(section: str, key: str) -> float:
        return mean([d[section][key] for d in details])

    return {
        "questions": len(details),
        "k": k,
        "configuration": {
            "embedding_model": container.embedder.name,
            "reranker": container.reranker.name,
            "vector_store": container.vector_store.backend,
            "llm": container.llm.model or "none (extractive)",
            "fusion": container.settings.fusion_method,
            "weights": {"semantic": container.settings.semantic_weight, "keyword": container.settings.keyword_weight},
        },
        "retrieval": {m: avg("retrieval", m) for m in ("recall_at_k", "precision_at_k", "mrr", "hit_rate", "context_relevance")},
        "generation": {
            m: avg("generation", m)
            for m in ("faithfulness", "answer_relevance", "key_fact_recall", "citation_precision", "citation_page_hit")
        },
        "details": details,
    }
