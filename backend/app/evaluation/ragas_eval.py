"""Optional RAGAS adapter (LLM-judged faithfulness, answer relevancy, context precision).

RAGAS is not a core dependency. Install ``requirements-optional.txt`` and
configure an LLM to use it; otherwise the built-in metrics in ``answer_eval``
and ``retrieval_eval`` are used on their own.
"""
from __future__ import annotations

from typing import Any

from app.config import Settings
from app.utils.errors import AppError


def ragas_available() -> bool:
    try:
        import datasets  # noqa: F401
        import langchain_openai  # noqa: F401
        import ragas  # noqa: F401
    except ImportError:
        return False
    return True


def run_ragas(samples: list[dict[str, Any]], settings: Settings) -> dict[str, float]:
    """Score ``samples`` with RAGAS using the configured OpenAI-compatible LLM as judge.

    Each sample needs ``question``, ``answer``, ``contexts`` (list of str) and ``ground_truth``.
    """
    if not ragas_available():
        raise AppError("RAGAS is not installed. Run `pip install -r backend/requirements-optional.txt`.")
    if settings.resolved_llm_provider == "none":
        raise AppError("RAGAS needs a judge model. Configure LLM_API_KEY (or LLM_BASE_URL for a local model).")

    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    judge = ChatOpenAI(
        model=settings.llm_model, api_key=settings.llm_api_key or "not-needed",
        base_url=settings.resolved_llm_base_url, temperature=0,
    )
    metrics = [faithfulness, context_precision]
    embeddings = None
    if settings.embedding_provider == "openai":
        # Answer relevancy needs an embedding model reachable through the same API style.
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.embedding_api_key or settings.llm_api_key,
            base_url=settings.embedding_base_url or None,
        )
        metrics.append(answer_relevancy)

    dataset = Dataset.from_list([
        {
            "question": s["question"], "answer": s["answer"],
            "contexts": s["contexts"], "ground_truth": s["ground_truth"],
        }
        for s in samples
    ])
    result = evaluate(dataset, metrics=metrics, llm=judge, embeddings=embeddings)
    frame = result.to_pandas()
    return {
        column: round(float(frame[column].mean()), 4)
        for column in frame.columns
        if column in ("faithfulness", "answer_relevancy", "context_precision")
    }
