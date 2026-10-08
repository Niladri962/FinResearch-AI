"""Evaluate the RAG pipeline against a labelled dataset.

Examples (from the repo root, with the backend virtualenv active):

    # Self-contained baseline: temp store, synthetic samples, offline models, no LLM
    python scripts/run_evaluation.py --offline

    # Same isolated run but with the embedding/reranker/LLM configured in .env
    python scripts/run_evaluation.py --isolated

    # Evaluate whatever is already indexed in your configured stores
    python scripts/run_evaluation.py --dataset evaluation/dataset.jsonl

    # Add LLM-judged RAGAS metrics (needs requirements-optional.txt and an LLM)
    python scripts/run_evaluation.py --isolated --ragas
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_sample_data import generate  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402
from app.evaluation.ragas_eval import run_ragas  # noqa: E402
from app.evaluation.runner import load_dataset, run_evaluation  # noqa: E402
from app.models.enums import DocumentStatus  # noqa: E402
from app.services.container import AppContainer  # noqa: E402
from app.utils.errors import AppError  # noqa: E402
from app.utils.logging import configure_logging  # noqa: E402


def ingest(container: AppContainer, paths: list[Path]) -> None:
    for path in paths:
        with path.open("rb") as handle:
            try:
                document = container.documents.save_upload(path.name, handle, {})
            except AppError as exc:
                print(f"  skipped {path.name}: {exc.message}")
                continue
        container.ingestion.run(document.id)
        result = container.documents.get(document.id)
        if result.status != DocumentStatus.READY.value:
            raise SystemExit(f"Ingestion failed for {path.name}: {result.error}")
        print(f"  indexed {path.name}: {result.chunk_count} chunks, {result.fact_count} facts")


def print_report(report: dict) -> None:
    config = report["configuration"]
    print(f"\nQuestions: {report['questions']}   k = {report['k']}")
    print(f"Embeddings: {config['embedding_model']}   Reranker: {config['reranker']}   LLM: {config['llm']}")
    print(f"Fusion: {config['fusion']} (semantic {config['weights']['semantic']}, keyword {config['weights']['keyword']})")
    for section, title in (("retrieval", "Retrieval"), ("generation", "Generation")):
        print(f"\n{title}")
        for metric, value in report[section].items():
            print(f"  {metric:<22} {value:.3f}")
    if "ragas" in report:
        print("\nRAGAS (LLM-judged)")
        for metric, value in report["ragas"].items():
            print(f"  {metric:<22} {value:.3f}")
    print("\nPer question (recall@k | MRR | faithfulness | key facts)")
    for detail in report["details"]:
        r, g = detail["retrieval"], detail["generation"]
        print(f"  {r['recall_at_k']:.0f} | {r['mrr']:.2f} | {g['faithfulness']:.2f} | {g['key_fact_recall']:.2f}  "
              f"[{detail['intent']}] {detail['question']}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=None, help="JSONL dataset (default: evaluation/dataset.jsonl)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--offline", action="store_true", help="isolated run with hash embeddings, lexical reranker and no LLM")
    parser.add_argument("--isolated", action="store_true", help="isolated temp store with the models configured in .env")
    parser.add_argument("--ragas", action="store_true", help="also compute LLM-judged RAGAS metrics")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "evaluation" / "results" / "latest.json")
    args = parser.parse_args()
    configure_logging("WARNING")

    dataset_path = args.dataset or REPO_ROOT / "evaluation" / "dataset.jsonl"
    if args.offline or args.isolated:
        workdir = Path(tempfile.mkdtemp(prefix="finresearch_eval_"))
        base = get_settings()
        overrides = dict(data_dir=workdir / "data", vector_store="memory", database_url="", redis_url="", qdrant_url="")
        if args.offline:
            overrides.update(embedding_provider="hash", reranker_provider="lexical", llm_provider="none")
        else:
            # Reuse already-downloaded models instead of fetching them into the temp directory.
            overrides["model_cache"] = str(base.model_cache_dir)
        settings = Settings(**{**base.model_dump(), **overrides})
        container = AppContainer(settings)
        print("Generating and indexing synthetic sample filings…")
        samples = generate(workdir / "samples", workdir / "dataset.jsonl")
        if args.dataset is None:
            dataset_path = workdir / "dataset.jsonl"
        ingest(container, samples)
    else:
        container = AppContainer(get_settings())

    rows = load_dataset(dataset_path)
    report = await run_evaluation(container, rows, k=args.k)

    if args.ragas:
        samples = [
            {"question": d["question"], "answer": d["answer"], "contexts": d["contexts"], "ground_truth": d["ground_truth"]}
            for d in report["details"]
        ]
        try:
            report["ragas"] = run_ragas(samples, container.settings)
        except AppError as exc:
            print(f"\nRAGAS skipped: {exc.message}")

    print_report(report)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    slim = {**report, "details": [{k: v for k, v in d.items() if k != "contexts"} for d in report["details"]]}
    args.out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out}")
    await container.aclose()


if __name__ == "__main__":
    asyncio.run(main())
