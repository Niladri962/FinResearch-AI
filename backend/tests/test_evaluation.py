"""RAG evaluation metrics and the end-to-end evaluation runner."""
from __future__ import annotations

import pytest

from app.evaluation.answer_eval import answer_relevance, citation_correctness, faithfulness, key_fact_recall
from app.evaluation.ragas_eval import ragas_available, run_ragas
from app.evaluation.retrieval_eval import (
    context_relevance,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.runner import load_dataset, run_evaluation
from app.models.schemas import Citation
from app.utils.errors import AppError
from tests.conftest import make_settings


class TestRetrievalMetrics:
    RELEVANCE = [False, True, False, True, False]

    def test_recall_at_k(self):
        assert recall_at_k(self.RELEVANCE, total_relevant=2, k=1) == 0.0
        assert recall_at_k(self.RELEVANCE, total_relevant=2, k=2) == 0.5
        assert recall_at_k(self.RELEVANCE, total_relevant=2, k=5) == 1.0
        assert recall_at_k(self.RELEVANCE, total_relevant=4, k=5) == 0.5
        assert recall_at_k([], total_relevant=0, k=5) == 0.0

    def test_precision_at_k(self):
        assert precision_at_k(self.RELEVANCE, 2) == 0.5
        assert precision_at_k(self.RELEVANCE, 5) == 0.4
        assert precision_at_k([True], 5) == 0.2        # fewer results than k still divides by k

    def test_mrr_and_hit_rate(self):
        assert reciprocal_rank(self.RELEVANCE) == 0.5
        assert reciprocal_rank([True]) == 1.0 and reciprocal_rank([False, False]) == 0.0
        assert hit_at_k(self.RELEVANCE, 1) == 0.0 and hit_at_k(self.RELEVANCE, 2) == 1.0

    def test_context_relevance(self):
        on_topic = "Interest rate risk: borrowings carry floating interest rates."
        off_topic = "The cafeteria menu changed."
        assert context_relevance("interest rate risk on borrowings", [on_topic]) == 1.0
        assert context_relevance("interest rate risk on borrowings", [on_topic, off_topic]) == 0.5
        assert context_relevance("interest rate risk", []) == 0.0


class TestGenerationMetrics:
    CONTEXT = ["Revenue from operations increased 12.8% to INR 10,600 crore, led by industrial components."]

    def test_faithfulness(self):
        assert faithfulness("Revenue from operations increased 12.8% to INR 10,600 crore [S1].", self.CONTEXT) == 1.0
        assert faithfulness("Revenue from operations increased 45% to INR 99,999 crore [S1].", self.CONTEXT) == 0.0
        mixed = "Revenue from operations increased 12.8% [S1].\nThe company acquired three overseas competitors last quarter."
        assert faithfulness(mixed, self.CONTEXT) == 0.5
        assert faithfulness("### Heading only", self.CONTEXT) == 0.0

    def test_answer_relevance_and_key_facts(self):
        assert answer_relevance("What was revenue growth?", "Revenue growth was 12.8%.") == 1.0
        assert answer_relevance("What was revenue growth?", "The weather was pleasant.") == 0.0
        assert key_fact_recall("Capex of INR 1,200 crore for the Gujarat complex.", ["1,200", "gujarat", "Pune"]) == pytest.approx(2 / 3)
        assert key_fact_recall("anything", []) == 0.0

    def test_citation_correctness(self):
        def cite(filename: str, page: int, end: int | None = None) -> Citation:
            return Citation(id="S1", chunk_id="c", document_id="d", document_title="t", filename=filename, page=page, page_end=end)

        assert citation_correctness([cite("ar.pdf", 4), cite("other.pdf", 2)], "ar.pdf", 4) == (0.5, 1.0)
        assert citation_correctness([cite("ar.pdf", 3, 5)], "ar.pdf", 4) == (1.0, 1.0)
        assert citation_correctness([cite("ar.pdf", 9)], "ar.pdf", 4) == (1.0, 0.0)
        assert citation_correctness([], "ar.pdf", 4) == (0.0, 0.0)


class TestRagasAdapter:
    def test_missing_dependency_or_judge_is_reported_clearly(self, tmp_path):
        with pytest.raises(AppError) as exc:
            run_ragas([], make_settings(tmp_path))
        expected = "judge model" if ragas_available() else "RAGAS is not installed"
        assert expected in exc.value.message


@pytest.mark.integration
class TestEvaluationRunner:
    def test_dataset_is_well_formed(self, sample_docs):
        rows = load_dataset(sample_docs["dataset"])
        assert len(rows) == 13
        names = {name for name in sample_docs if name != "dataset"}
        for row in rows:
            assert row.question.endswith("?") and row.expected_answer and row.key_facts
            assert row.relevant_document in names and row.relevant_page >= 1

    def test_invalid_dataset_row_is_reported_with_its_line(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"question": "q"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
            load_dataset(path)

    async def test_end_to_end_scores(self, client, sample_docs):
        rows = load_dataset(sample_docs["dataset"])
        report = await run_evaluation(client.app.state.container, rows, k=5)
        assert report["questions"] == 13 and report["k"] == 5
        assert report["configuration"]["llm"] == "none (extractive)"
        retrieval, generation = report["retrieval"], report["generation"]
        # Offline stack (hash embeddings + lexical reranker) on the synthetic samples. These floors
        # catch regressions in parsing, chunking, fusion, the statement lane or routing.
        assert retrieval["hit_rate"] >= 0.9 and retrieval["recall_at_k"] >= 0.9
        assert retrieval["mrr"] >= 0.6 and retrieval["context_relevance"] >= 0.3
        assert generation["faithfulness"] >= 0.9          # extractive answers quote the sources
        assert generation["citation_precision"] >= 0.5
        assert all(set(d) >= {"question", "intent", "retrieval", "generation", "answer"} for d in report["details"])
