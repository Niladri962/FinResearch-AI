"use client";

import { ArrowLeft, Download, FilePlus2, Printer, Trash2 } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

import { Markdown } from "@/components/Markdown";
import { SourcePanel } from "@/components/SourcePanel";
import { Badge, Card, CompanySelect, EmptyState, ErrorBanner, PageHeader, Spinner } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { api, ApiError } from "@/lib/api";
import { relativeTime } from "@/lib/format";
import type { ReportDetail } from "@/lib/types";

function ReportsView() {
  const params = useSearchParams();
  const companies = useApi(api.companies);
  const reports = useApi(api.reports);
  const [companyId, setCompanyId] = useState<number | null>(params.get("company") ? Number(params.get("company")) : null);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<ReportDetail | null>(null);
  const [preview, setPreview] = useState<{ chunkId: string; label: string } | null>(null);

  useEffect(() => {
    if (companyId === null && companies.data?.length) setCompanyId(companies.data[0].id);
  }, [companies.data, companyId]);

  const generate = async () => {
    if (companyId === null) return;
    setGenerating(true);
    setError(null);
    try {
      setOpen(await api.generateReport(companyId));
      reports.reload();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Report generation failed.");
    } finally {
      setGenerating(false);
    }
  };

  const view = async (id: string) => {
    try {
      setOpen(await api.report(id));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not open the report.");
    }
  };

  const remove = async (id: string) => {
    if (!window.confirm("Delete this report?")) return;
    await api.deleteReport(id).catch(() => undefined);
    reports.reload();
  };

  const download = (report: ReportDetail) => {
    const url = URL.createObjectURL(new Blob([report.content], { type: "text/markdown;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `${report.title.replace(/[^\w]+/g, "-").replace(/^-|-$/g, "").toLowerCase()}.md`;
    link.click();
    URL.revokeObjectURL(url);
  };

  if (open) {
    return (
      <>
        <div className="no-print mb-5 flex flex-wrap items-center justify-between gap-2">
          <button onClick={() => setOpen(null)} className="btn-ghost">
            <ArrowLeft className="h-4 w-4" aria-hidden /> All reports
          </button>
          <div className="flex gap-2">
            <button onClick={() => window.print()} className="btn-ghost">
              <Printer className="h-4 w-4" aria-hidden /> Print / PDF
            </button>
            <button onClick={() => download(open)} className="btn-primary">
              <Download className="h-4 w-4" aria-hidden /> Markdown
            </button>
          </div>
        </div>
        <article className="card px-6 py-7 sm:px-10">
          <Markdown
            content={open.content}
            onCite={(id) => {
              const source = open.sources.find((s) => s.id === id);
              if (source) setPreview({ chunkId: source.chunk_id, label: id });
            }}
          />
        </article>
        <SourcePanel chunkId={preview?.chunkId ?? null} label={preview?.label} onClose={() => setPreview(null)} />
      </>
    );
  }

  return (
    <>
      <PageHeader title="Reports" subtitle="Generate a structured research report. Tables come from the financial engine; narrative is written from retrieved passages and every claim links to its source." />
      {companies.error && <ErrorBanner message={companies.error} onRetry={companies.reload} />}

      {companies.data && companies.data.length > 0 && (
        <Card>
          <div className="flex flex-wrap items-end gap-3">
            <CompanySelect companies={companies.data} value={companyId} onChange={setCompanyId} />
            <button onClick={() => void generate()} disabled={generating || companyId === null} className="btn-primary">
              <FilePlus2 className="h-4 w-4" aria-hidden />
              {generating ? "Generating…" : "Generate report"}
            </button>
          </div>
          {generating && (
            <div className="mt-4">
              <Spinner label="Computing ratios, retrieving evidence and writing 13 sections — this can take up to a minute…" />
            </div>
          )}
          {error && <div className="mt-4"><ErrorBanner message={error} /></div>}
        </Card>
      )}

      <div className="mt-6">
        {reports.loading && <Spinner />}
        {reports.error && <ErrorBanner message={reports.error} onRetry={reports.reload} />}
        {reports.data && !reports.data.length && (
          <EmptyState
            title="No reports yet"
            body="A report covers revenue, profitability, balance sheet, cash flow, management commentary, risks, growth opportunities, key ratios and historical trends."
          />
        )}
        {reports.data && reports.data.length > 0 && (
          <ul className="card divide-y divide-line">
            {reports.data.map((report) => (
              <li key={report.id} className="flex items-center justify-between gap-3 px-5 py-3.5">
                <button onClick={() => void view(report.id)} className="min-w-0 text-left">
                  <p className="truncate text-sm font-medium text-ink hover:text-accent">{report.title}</p>
                  <p className="mt-0.5 text-xs text-ink-muted">
                    {relativeTime(report.created_at)}
                    {report.meta.periods?.length ? ` · ${report.meta.periods[0]}–${report.meta.periods[report.meta.periods.length - 1]}` : ""}
                    {report.meta.source_count !== undefined ? ` · ${report.meta.source_count} sources` : ""}
                  </p>
                </button>
                <div className="flex shrink-0 items-center gap-2">
                  {report.meta.mode && <Badge tone={report.meta.mode === "generative" ? "accent" : "neutral"}>{report.meta.mode === "generative" ? "LLM narrative" : "Extractive"}</Badge>}
                  <button onClick={() => void remove(report.id)} aria-label={`Delete report ${report.title}`} className="rounded-lg p-1.5 text-ink-muted hover:bg-page hover:text-critical">
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </>
  );
}

export default function ReportsPage() {
  return (
    <Suspense>
      <ReportsView />
    </Suspense>
  );
}
