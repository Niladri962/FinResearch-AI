"use client";

import { ArrowRight, Building2, Database, FileText, MessagesSquare } from "lucide-react";
import Link from "next/link";

import { RiskFlagList, StatTile } from "@/components/Artifacts";
import { DataChart } from "@/components/DataChart";
import { Badge, Card, EmptyState, ErrorBanner, PageHeader, Spinner } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { relativeTime, titleCase } from "@/lib/format";

function CountTile({ icon: Icon, label, value, detail }: { icon: typeof FileText; label: string; value: number; detail?: string }) {
  return (
    <div className="card flex items-center gap-3 px-4 py-3.5">
      <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent-soft text-accent-ink">
        <Icon className="h-4 w-4" aria-hidden />
      </span>
      <div>
        <p className="text-xs text-ink-soft">{label}</p>
        <p className="text-lg font-semibold leading-tight text-ink">{value.toLocaleString()}</p>
        {detail && <p className="text-[11px] text-ink-muted">{detail}</p>}
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const { data, error, loading, reload } = useApi(api.dashboard);

  if (loading) return <Spinner label="Loading dashboard…" />;
  if (error || !data) return <ErrorBanner message={error || "No data."} onRetry={reload} />;

  const spotlight = data.spotlight;
  const performance = data.performance;

  return (
    <>
      <PageHeader title="Dashboard" subtitle="AI-powered financial research and analysis across your uploaded filings." />

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <CountTile icon={Building2} label="Companies analysed" value={data.companies} />
        <CountTile
          icon={FileText}
          label="Documents indexed"
          value={data.documents_ready}
          detail={[data.documents_processing && `${data.documents_processing} processing`, data.documents_failed && `${data.documents_failed} failed`].filter(Boolean).join(" · ") || undefined}
        />
        <CountTile icon={Database} label="Financial facts extracted" value={data.facts} detail={`${data.chunks.toLocaleString()} passages`} />
        <CountTile icon={MessagesSquare} label="Research threads" value={data.conversations} detail={`${data.reports} report${data.reports === 1 ? "" : "s"}`} />
      </div>

      {!data.documents && (
        <div className="mt-6">
          <EmptyState
            title="Upload your first filing"
            body="Add annual reports, quarterly results, earnings-call transcripts or financial statements. FinResearch AI indexes them, extracts the financial tables and answers questions with page-level citations."
            action={<Link href="/documents" className="btn-primary">Go to Documents</Link>}
          />
        </div>
      )}

      {spotlight && spotlight.kpis.length > 0 && (
        <section className="mt-8">
          <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold text-ink">Key financial metrics — {spotlight.company.name}</h2>
              <p className="text-xs text-ink-muted">
                {spotlight.kpis[0].period} · calculated from the extracted statements
                {spotlight.company.currency ? ` · ${spotlight.company.currency} ${spotlight.company.unit ?? ""}` : ""}
              </p>
            </div>
            <Link href={`/analysis?company=${spotlight.company.id}`} className="flex items-center gap-1 text-sm font-medium text-accent hover:underline">
              Full analysis <ArrowRight className="h-3.5 w-3.5" aria-hidden />
            </Link>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {spotlight.kpis.map((kpi) => (
              <StatTile key={kpi.key} calc={kpi} />
            ))}
          </div>
          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            {spotlight.charts.slice(0, 4).map((chart) => (
              <div key={chart.id} className="card p-5">
                <DataChart chart={chart} />
              </div>
            ))}
          </div>
          {spotlight.risk_flags.length > 0 && (
            <Card title="Risk signals" subtitle="Rule-based checks on the latest reported figures" className="mt-4">
              <RiskFlagList flags={spotlight.risk_flags.slice(0, 5)} />
            </Card>
          )}
        </section>
      )}

      <div className="mt-8 grid gap-4 lg:grid-cols-2">
        <Card title="Recent questions">
          {data.recent_questions.length ? (
            <ul className="divide-y divide-line">
              {data.recent_questions.map((item) => (
                <li key={item.conversation_id + item.asked_at} className="flex items-baseline justify-between gap-3 py-2 first:pt-0 last:pb-0">
                  <Link href={`/chat?q=${encodeURIComponent(item.question)}`} className="min-w-0 truncate text-sm text-ink hover:text-accent">
                    {item.question}
                  </Link>
                  <span className="shrink-0 text-xs text-ink-muted">{relativeTime(item.asked_at)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-ink-soft">Questions you ask in Research Chat will appear here.</p>
          )}
        </Card>

        <Card title="Recent documents" actions={<Link href="/documents" className="text-xs font-medium text-accent hover:underline">All documents</Link>}>
          {data.recent_documents.length ? (
            <ul className="divide-y divide-line">
              {data.recent_documents.map((doc) => (
                <li key={doc.id} className="flex items-center justify-between gap-3 py-2 first:pt-0 last:pb-0">
                  <div className="min-w-0">
                    <p className="truncate text-sm text-ink">{doc.company_name ? `${doc.company_name} — ${doc.title}` : doc.filename}</p>
                    <p className="text-xs text-ink-muted">{relativeTime(doc.uploaded_at)}</p>
                  </div>
                  <Badge tone={doc.status === "ready" ? "good" : doc.status === "failed" ? "critical" : "warn"}>{titleCase(doc.status)}</Badge>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-ink-soft">No documents yet.</p>
          )}
        </Card>

        <Card title="Recent research">
          {data.recent_research.length ? (
            <ul className="divide-y divide-line">
              {data.recent_research.map((conversation) => (
                <li key={conversation.id} className="flex items-baseline justify-between gap-3 py-2 first:pt-0 last:pb-0">
                  <Link href={`/chat?c=${conversation.id}`} className="min-w-0 truncate text-sm text-ink hover:text-accent">
                    {conversation.title}
                  </Link>
                  <span className="shrink-0 text-xs text-ink-muted">
                    {conversation.message_count} messages · {relativeTime(conversation.updated_at)}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-ink-soft">No research threads yet.</p>
          )}
        </Card>

        <Card title="Pipeline performance" subtitle={`Last ${performance.queries ?? 0} queries`}>
          {performance.queries ? (
            <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
              {[
                ["Avg response time", performance.avg_total_ms != null ? `${(performance.avg_total_ms / 1000).toFixed(2)} s` : "—"],
                ["Avg retrieval", performance.avg_retrieval_ms != null ? `${performance.avg_retrieval_ms.toFixed(0)} ms` : "—"],
                ["Avg LLM time", performance.avg_llm_ms != null ? `${(performance.avg_llm_ms / 1000).toFixed(2)} s` : "No LLM calls"],
                ["Avg grounding score", performance.avg_grounding_score != null ? `${(performance.avg_grounding_score * 100).toFixed(0)}%` : "—"],
                ["Tokens used", (performance.total_tokens ?? 0).toLocaleString()],
                ["Errors", String(performance.errors ?? 0)],
              ].map(([label, value]) => (
                <div key={label}>
                  <dt className="text-xs text-ink-soft">{label}</dt>
                  <dd className="font-medium tabular-nums text-ink">{value}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="text-sm text-ink-soft">Latency, token usage and grounding scores appear after the first query.</p>
          )}
        </Card>
      </div>
    </>
  );
}
