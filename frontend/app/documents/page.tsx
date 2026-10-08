"use client";

import { ChevronDown, Trash2, UploadCloud } from "lucide-react";
import { useRef, useState } from "react";

import { Badge, Card, EmptyState, ErrorBanner, PageHeader, Spinner } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { api, ApiError } from "@/lib/api";
import { fileSize, relativeTime, titleCase } from "@/lib/format";
import type { DocumentItem } from "@/lib/types";

const ACCEPT = ".pdf,.docx,.xlsx,.txt,.md";
const TYPES = [
  ["", "Detect automatically"],
  ["annual_report", "Annual report"],
  ["quarterly_report", "Quarterly report"],
  ["earnings_call_transcript", "Earnings call transcript"],
  ["investor_presentation", "Investor presentation"],
  ["financial_statement", "Financial statements"],
  ["management_discussion", "Management discussion & analysis"],
  ["research_report", "Research report"],
  ["company_filing", "Company filing"],
];

const STATUS_TONE = { ready: "good", failed: "critical", processing: "warn", queued: "warn" } as const;

export default function DocumentsPage() {
  const [pollMs, setPollMs] = useState<number | null>(null);
  const { data, error, loading, reload } = useApi(
    async () => {
      const documents = await api.documents();
      // Poll only while something is still being processed.
      setPollMs(documents.some((d) => d.status === "queued" || d.status === "processing") ? 2000 : null);
      return documents;
    },
    [],
    pollMs,
  );
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [messages, setMessages] = useState<{ ok: boolean; text: string }[]>([]);
  const [showOptions, setShowOptions] = useState(false);
  const [meta, setMeta] = useState({ company: "", fiscal_year: "", quarter: "", document_type: "" });

  const upload = async (files: File[]) => {
    if (!files.length || uploading) return;
    setUploading(true);
    setMessages([]);
    try {
      const response = await api.upload(files, meta);
      setMessages(
        response.results.map((r) =>
          r.error ? { ok: false, text: `${r.filename}: ${r.error.message}` } : { ok: true, text: `${r.filename}: accepted, processing started.` },
        ),
      );
    } catch (err) {
      setMessages([{ ok: false, text: err instanceof ApiError ? err.message : "Upload failed." }]);
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
      reload();
    }
  };

  const remove = async (doc: DocumentItem) => {
    if (!window.confirm(`Delete "${doc.filename}"? Its passages and extracted figures are removed from the index.`)) return;
    try {
      await api.deleteDocument(doc.id);
    } catch (err) {
      setMessages([{ ok: false, text: err instanceof ApiError ? err.message : "Delete failed." }]);
    }
    reload();
  };

  return (
    <>
      <PageHeader title="Documents" subtitle="Annual and quarterly reports, investor presentations, earnings-call transcripts, financial statements and filings." />

      <Card>
        <div
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            void upload(Array.from(event.dataTransfer.files));
          }}
          className={`flex flex-col items-center rounded-xl border-2 border-dashed px-6 py-9 text-center transition-colors ${dragging ? "border-accent bg-accent-soft" : "border-line"}`}
        >
          <UploadCloud className="h-7 w-7 text-ink-muted" aria-hidden />
          <p className="mt-3 text-sm font-medium text-ink">Drop files here, or</p>
          <button onClick={() => inputRef.current?.click()} disabled={uploading} className="btn-primary mt-3">
            {uploading ? "Uploading…" : "Choose files"}
          </button>
          <input ref={inputRef} type="file" multiple accept={ACCEPT} className="hidden" onChange={(event) => void upload(Array.from(event.target.files || []))} />
          <p className="mt-3 text-xs text-ink-muted">PDF, DOCX, XLSX, TXT or MD · up to 10 files per upload</p>
        </div>

        <button onClick={() => setShowOptions((v) => !v)} className="mt-4 flex items-center gap-1 text-xs font-medium text-ink-soft hover:text-ink" aria-expanded={showOptions}>
          <ChevronDown className={`h-3.5 w-3.5 transition-transform ${showOptions ? "rotate-180" : ""}`} aria-hidden />
          Set company, period or type manually
        </button>
        {showOptions && (
          <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <label>
              <span className="label">Company</span>
              <input className="input" placeholder="Detected from the document" value={meta.company} onChange={(e) => setMeta({ ...meta, company: e.target.value })} />
            </label>
            <label>
              <span className="label">Fiscal year</span>
              <input className="input" inputMode="numeric" placeholder="e.g. 2025" value={meta.fiscal_year} onChange={(e) => setMeta({ ...meta, fiscal_year: e.target.value.replace(/\D/g, "").slice(0, 4) })} />
            </label>
            <label>
              <span className="label">Quarter</span>
              <select className="input" value={meta.quarter} onChange={(e) => setMeta({ ...meta, quarter: e.target.value })}>
                <option value="">Full year / detect</option>
                {[1, 2, 3, 4].map((q) => <option key={q} value={q}>Q{q}</option>)}
              </select>
            </label>
            <label>
              <span className="label">Document type</span>
              <select className="input" value={meta.document_type} onChange={(e) => setMeta({ ...meta, document_type: e.target.value })}>
                {TYPES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
            <p className="text-xs text-ink-muted sm:col-span-2 lg:col-span-4">These values apply to every file in the next upload and override automatic detection.</p>
          </div>
        )}

        {messages.length > 0 && (
          <ul className="mt-4 space-y-1.5" aria-live="polite">
            {messages.map((message) => (
              <li key={message.text} className="flex items-start gap-2 text-sm text-ink">
                <Badge tone={message.ok ? "good" : "critical"}>{message.ok ? "Accepted" : "Rejected"}</Badge>
                <span>{message.text}</span>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <div className="mt-6">
        {loading && <Spinner />}
        {error && <ErrorBanner message={error} onRetry={reload} />}
        {data && !data.length && <EmptyState title="No documents indexed" body="Uploaded documents appear here with their detected company, period and processing status." />}
        {data && data.length > 0 && (
          <div className="card overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-line text-xs text-ink-soft">
                  {["Document", "Company", "Type", "Period", "Indexed", "Status", ""].map((heading) => (
                    <th key={heading} className="whitespace-nowrap px-4 py-3 font-medium">{heading}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.map((doc) => (
                  <tr key={doc.id} className="border-b border-line align-top last:border-b-0">
                    <td className="max-w-[18rem] px-4 py-3">
                      <p className="truncate font-medium text-ink" title={doc.filename}>{doc.filename}</p>
                      <p className="text-xs text-ink-muted">{fileSize(doc.size_bytes)} · {relativeTime(doc.uploaded_at)}</p>
                      {doc.error && <p className="mt-1 text-xs text-ink-soft">{doc.error}</p>}
                      {doc.warnings.map((warning) => <p key={warning} className="mt-1 text-xs text-ink-muted">{warning}</p>)}
                    </td>
                    <td className="px-4 py-3 text-ink-soft">{doc.company_name || "—"}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-ink-soft">{doc.status === "ready" ? doc.document_type_label : "—"}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-ink-soft">{doc.reporting_period || "—"}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-xs text-ink-soft">
                      {doc.status === "ready" ? (
                        <>
                          {doc.page_count} pages · {doc.chunk_count} passages
                          <br />
                          {doc.fact_count} financial facts
                        </>
                      ) : "—"}
                    </td>
                    <td className="px-4 py-3">
                      <Badge tone={STATUS_TONE[doc.status]}>{titleCase(doc.status)}</Badge>
                      {doc.status !== "ready" && doc.status !== "failed" && (
                        <p className="mt-1 max-w-[11rem] text-xs text-ink-muted">
                          {doc.progress || "Waiting to start"}. Large reports can take a few minutes.
                        </p>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button onClick={() => void remove(doc)} aria-label={`Delete ${doc.filename}`} className="rounded-lg p-1.5 text-ink-muted hover:bg-page hover:text-critical">
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
