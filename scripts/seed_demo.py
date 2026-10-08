"""Load the synthetic sample filings into a running FinResearch AI backend.

    python scripts/seed_demo.py                          # http://localhost:8000
    python scripts/seed_demo.py --api http://host:8000 --api-key <key>

The samples describe fictitious companies (see scripts/generate_sample_data.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_sample_data import generate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--samples", type=Path, default=REPO_ROOT / "data" / "samples")
    args = parser.parse_args()

    paths = sorted(args.samples.glob("*.pdf"))
    if not paths:
        paths = generate(args.samples, REPO_ROOT / "evaluation" / "dataset.jsonl")
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    with httpx.Client(base_url=args.api.rstrip("/"), headers=headers, timeout=120) as client:
        try:
            client.get("/api/health").raise_for_status()
        except httpx.HTTPError as exc:
            raise SystemExit(f"Backend not reachable at {args.api}: {exc}") from exc

        pending: dict[str, str] = {}
        for path in paths:
            with path.open("rb") as handle:
                response = client.post("/api/documents/upload", files={"files": (path.name, handle, "application/pdf")})
            if response.status_code == 409:
                print(f"  already uploaded: {path.name}")
                continue
            if response.status_code >= 400:
                raise SystemExit(f"Upload failed for {path.name}: {response.text}")
            pending[response.json()["results"][0]["document"]["id"]] = path.name
            print(f"  uploaded: {path.name}")

        deadline = time.time() + 600
        while pending and time.time() < deadline:
            time.sleep(2)
            for document_id in list(pending):
                document = client.get(f"/api/documents/{document_id}").json()
                if document["status"] in ("ready", "failed"):
                    detail = f"{document['chunk_count']} passages, {document['fact_count']} facts" if document["status"] == "ready" else document["error"]
                    print(f"  {document['status']}: {pending.pop(document_id)} ({detail})")
        if pending:
            raise SystemExit("Timed out waiting for: " + ", ".join(pending.values()))
    print("Done. Open the app and try: \"Why did Aurora's operating margin decline?\"")


if __name__ == "__main__":
    main()
