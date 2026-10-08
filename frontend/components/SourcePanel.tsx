"use client";

import { FileText, X } from "lucide-react";
import { useEffect, useState } from "react";

import { api, ApiError } from "@/lib/api";
import { titleCase } from "@/lib/format";
import type { ChunkPreview } from "@/lib/types";

import { Markdown } from "./Markdown";
import { Badge, ErrorBanner, Spinner } from "./ui";

/** Slide-over showing the full passage behind a citation. */
export function SourcePanel({ chunkId, label, onClose }: { chunkId: string | null; label?: string; onClose: () => void }) {
  const [chunk, setChunk] = useState<ChunkPreview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!chunkId) return;
    let cancelled = false;
    setChunk(null);
    setError(null);
    api
      .chunk(chunkId)
      .then((data) => !cancelled && setChunk(data))
      .catch((err) => !cancelled && setError(err instanceof ApiError ? err.message : "Could not load the source."));
    return () => {
      cancelled = true;
    };
  }, [chunkId]);

  useEffect(() => {
    if (!chunkId) return;
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [chunkId, onClose]);

  if (!chunkId) return null;
  const pages = chunk && (chunk.page_end !== chunk.page_start ? `Pages ${chunk.page_start}–${chunk.page_end}` : `Page ${chunk.page_start}`);

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true" aria-label="Source preview">
      <div className="absolute inset-0 bg-black/30" onClick={onClose} />
      <aside className="relative flex h-full w-full max-w-xl flex-col border-l border-line bg-surface">
        <header className="flex items-start gap-3 border-b border-line px-5 py-4">
          <FileText className="mt-0.5 h-4 w-4 shrink-0 text-ink-soft" aria-hidden />
          <div className="min-w-0 flex-1">
            <p className="text-sm font-semibold text-ink">
              {label && <span className="mr-2 font-mono text-xs text-accent-ink">{label}</span>}
              {chunk ? chunk.document_title : "Source"}
            </p>
            {chunk && (
              <p className="mt-0.5 text-xs text-ink-soft">
                {[chunk.company_name, pages, chunk.section].filter(Boolean).join(" · ")}
              </p>
            )}
          </div>
          <button onClick={onClose} aria-label="Close source preview" className="rounded-lg p-1.5 text-ink-soft hover:bg-page">
            <X className="h-4 w-4" />
          </button>
        </header>
        <div className="flex-1 overflow-y-auto px-5 py-4">
          {error && <ErrorBanner message={error} />}
          {!chunk && !error && <Spinner label="Loading passage…" />}
          {chunk && (
            <>
              <div className="mb-3">
                <Badge tone="accent">{titleCase(chunk.chunk_type)}</Badge>
              </div>
              <Markdown content={chunk.text} />
              <p className="mt-6 border-t border-line pt-3 text-xs text-ink-muted">
                Passage as extracted from the uploaded document. Chunk id {chunk.id}.
              </p>
            </>
          )}
        </div>
      </aside>
    </div>
  );
}
