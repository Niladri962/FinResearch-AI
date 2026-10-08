"use client";

import { LineChart, MessagesSquare, ScrollText } from "lucide-react";
import Link from "next/link";

import { Badge, EmptyState, ErrorBanner, PageHeader, Spinner } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { api } from "@/lib/api";

export default function CompaniesPage() {
  const { data, error, loading, reload } = useApi(api.companies);

  return (
    <>
      <PageHeader title="Companies" subtitle="Companies are detected automatically from the documents you upload." />
      {loading && <Spinner />}
      {error && <ErrorBanner message={error} onRetry={reload} />}
      {data && !data.length && (
        <EmptyState
          title="No companies yet"
          body="Upload a filing and the company, reporting period and document type are identified for you."
          action={<Link href="/documents" className="btn-primary">Upload documents</Link>}
        />
      )}
      <div className="grid gap-4 md:grid-cols-2">
        {data?.map((company) => (
          <article key={company.id} className="card p-5">
            <h2 className="text-base font-semibold text-ink">{company.name}</h2>
            <p className="mt-1 text-xs text-ink-muted">
              {company.document_count} document{company.document_count === 1 ? "" : "s"}
              {company.currency ? ` · reports in ${company.currency}${company.unit ? ` ${company.unit}` : ""}` : ""}
            </p>

            <dl className="mt-4 space-y-3 text-sm">
              <div>
                <dt className="label">Documents cover</dt>
                <dd className="flex flex-wrap gap-1.5">
                  {company.fiscal_years.length ? company.fiscal_years.map((year) => <Badge key={year}>FY{year}</Badge>) : <span className="text-ink-soft">Period not detected</span>}
                </dd>
              </div>
              <div>
                <dt className="label">Structured financial data</dt>
                <dd className="flex flex-wrap gap-1.5">
                  {company.periods_with_data.length ? (
                    company.periods_with_data.map((period) => <Badge key={period} tone="accent">{period}</Badge>)
                  ) : (
                    <span className="text-ink-soft">None extracted — ratios and charts are unavailable, but questions still work.</span>
                  )}
                </dd>
              </div>
            </dl>

            <div className="mt-5 flex flex-wrap gap-2">
              <Link href={`/analysis?company=${company.id}`} className="btn-ghost">
                <LineChart className="h-4 w-4" aria-hidden /> Analysis
              </Link>
              <Link href={`/chat?q=${encodeURIComponent(`Summarize ${company.name}'s financial performance.`)}`} className="btn-ghost">
                <MessagesSquare className="h-4 w-4" aria-hidden /> Ask
              </Link>
              <Link href={`/reports?company=${company.id}`} className="btn-ghost">
                <ScrollText className="h-4 w-4" aria-hidden /> Report
              </Link>
            </div>
          </article>
        ))}
      </div>
    </>
  );
}
