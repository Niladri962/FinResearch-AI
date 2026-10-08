"use client";

import { GitCompareArrows } from "lucide-react";
import { useState } from "react";

import { TableView } from "@/components/Artifacts";
import { DataChart } from "@/components/DataChart";
import { Markdown } from "@/components/Markdown";
import { Card, Disclaimer, EmptyState, ErrorBanner, PageHeader, Spinner } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { api, ApiError } from "@/lib/api";
import type { CompareResponse } from "@/lib/types";

export default function ComparePage() {
  const companies = useApi(api.companies);
  const [mode, setMode] = useState<"company" | "period">("company");
  const [selected, setSelected] = useState<number[]>([]);
  const [periods, setPeriods] = useState<string[]>([]);
  const [withNarrative, setWithNarrative] = useState(true);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const list = companies.data || [];
  const single = list.find((c) => c.id === selected[0]);
  const ready = mode === "company" ? selected.length >= 2 : selected.length === 1 && (periods.length === 0 || periods.length >= 2);

  const switchMode = (next: "company" | "period") => {
    setMode(next);
    setSelected([]);
    setPeriods([]);
    setResult(null);
    setError(null);
  };

  const toggleCompany = (id: number) => {
    setResult(null);
    if (mode === "period") {
      setSelected([id]);
      setPeriods([]);
    } else {
      setSelected((ids) => (ids.includes(id) ? ids.filter((v) => v !== id) : ids.length < 4 ? [...ids, id] : ids));
    }
  };

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      setResult(await api.compare({ company_ids: selected, periods: mode === "period" ? periods : [], include_narrative: withNarrative }));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Comparison failed.");
    } finally {
      setLoading(false);
    }
  };

  const chip = (active: boolean) =>
    `rounded-full border px-3 py-1.5 text-sm transition-colors ${
      active ? "border-accent bg-accent-soft font-medium text-accent-ink" : "border-line bg-raised text-ink-soft hover:border-accent"
    }`;

  return (
    <>
      <PageHeader title="Comparisons" subtitle="Compare companies side by side, or one company across reporting periods. Missing values are shown as “Not available”, never estimated." />
      {companies.loading && <Spinner />}
      {companies.error && <ErrorBanner message={companies.error} onRetry={companies.reload} />}
      {companies.data && !list.length && <EmptyState title="No companies yet" body="Upload documents for at least one company to compare periods, or two to compare companies." />}

      {list.length > 0 && (
        <Card>
          <div className="mb-4 inline-flex rounded-lg border border-line p-0.5" role="tablist" aria-label="Comparison type">
            {(["company", "period"] as const).map((value) => (
              <button
                key={value}
                role="tab"
                aria-selected={mode === value}
                onClick={() => switchMode(value)}
                className={`rounded-md px-3 py-1.5 text-sm ${mode === value ? "bg-accent text-white" : "text-ink-soft hover:text-ink"}`}
              >
                {value === "company" ? "Company vs company" : "Period vs period"}
              </button>
            ))}
          </div>

          <p className="label">{mode === "company" ? "Choose two to four companies" : "Choose a company"}</p>
          <div className="flex flex-wrap gap-2">
            {list.map((company) => (
              <button key={company.id} onClick={() => toggleCompany(company.id)} aria-pressed={selected.includes(company.id)} className={chip(selected.includes(company.id))}>
                {company.name}
              </button>
            ))}
          </div>
          {mode === "company" && list.length < 2 && <p className="mt-2 text-xs text-ink-muted">Upload documents for a second company to enable company comparison.</p>}

          {mode === "period" && single && (
            <div className="mt-4">
              <p className="label">Periods (leave empty for the latest two, or pick at least two)</p>
              {single.periods_with_data.length ? (
                <div className="flex flex-wrap gap-2">
                  {single.periods_with_data.map((period) => (
                    <button
                      key={period}
                      onClick={() => setPeriods((current) => (current.includes(period) ? current.filter((p) => p !== period) : [...current, period]))}
                      aria-pressed={periods.includes(period)}
                      className={chip(periods.includes(period))}
                    >
                      {period}
                    </button>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-ink-soft">No structured financial data was extracted for this company.</p>
              )}
            </div>
          )}

          <div className="mt-5 flex flex-wrap items-center gap-4">
            <button onClick={() => void run()} disabled={!ready || loading} className="btn-primary">
              <GitCompareArrows className="h-4 w-4" aria-hidden />
              {loading ? "Comparing…" : "Compare"}
            </button>
            <label className="flex items-center gap-2 text-sm text-ink-soft">
              <input type="checkbox" checked={withNarrative} onChange={(event) => setWithNarrative(event.target.checked)} />
              Include qualitative analysis from the documents
            </label>
          </div>
        </Card>
      )}

      {error && <div className="mt-4"><ErrorBanner message={error} /></div>}
      {loading && <div className="mt-6"><Spinner label="Calculating and retrieving evidence…" /></div>}

      {result && !loading && (
        <div className="mt-6 space-y-6">
          <Card title="Numerical comparison" subtitle="Calculated from the extracted statements">
            <TableView table={result.table} />
          </Card>
          {result.charts.map((chart) => (
            <div key={chart.id} className="card p-5">
              <DataChart chart={chart} height={280} />
            </div>
          ))}
          {result.narrative && (
            <Card title="Qualitative analysis" subtitle="From retrieved passages, with citations">
              <Markdown content={result.narrative} />
            </Card>
          )}
          <Disclaimer text={result.disclaimer} />
        </div>
      )}
    </>
  );
}
