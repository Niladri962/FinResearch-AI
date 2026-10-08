"use client";

import { Sparkles } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

import { CalculationCard, RiskFlagList, StatTile, TableView } from "@/components/Artifacts";
import { DataChart } from "@/components/DataChart";
import { Markdown } from "@/components/Markdown";
import { SourcePanel } from "@/components/SourcePanel";
import { Card, CompanySelect, Disclaimer, EmptyState, ErrorBanner, PageHeader, Spinner } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { api, ApiError } from "@/lib/api";
import type { AnalyzeResponse } from "@/lib/types";

const CATEGORIES = ["liquidity", "solvency", "profitability", "efficiency", "growth"] as const;

function AnalysisView() {
  const params = useSearchParams();
  const companies = useApi(api.companies);
  const [companyId, setCompanyId] = useState<number | null>(params.get("company") ? Number(params.get("company")) : null);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [projection, setProjection] = useState(false);
  const [narrative, setNarrative] = useState<{ text: string; loading: boolean } | null>(null);
  const [category, setCategory] = useState<(typeof CATEGORIES)[number]>("profitability");
  const [preview, setPreview] = useState<string | null>(null);

  useEffect(() => {
    if (companyId === null && companies.data?.length) setCompanyId(companies.data[0].id);
  }, [companies.data, companyId]);

  useEffect(() => {
    if (companyId === null) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setNarrative(null);
    api
      .analyze({ company_id: companyId, include_projection: projection })
      .then((data) => !cancelled && setAnalysis(data))
      .catch((err) => !cancelled && setError(err instanceof ApiError ? err.message : "Analysis failed."))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [companyId, projection]);

  const writeNarrative = async () => {
    if (companyId === null) return;
    setNarrative({ text: "", loading: true });
    try {
      const data = await api.analyze({ company_id: companyId, include_narrative: true });
      setNarrative({ text: data.narrative || "No commentary could be generated.", loading: false });
    } catch (err) {
      setNarrative({ text: err instanceof ApiError ? err.message : "Commentary failed.", loading: false });
    }
  };

  return (
    <>
      <PageHeader
        title="Financial Analysis"
        subtitle="Metrics, ratios and trends calculated from the statements extracted from your documents."
        actions={
          companies.data?.length ? <CompanySelect companies={companies.data} value={companyId} onChange={setCompanyId} label="" /> : undefined
        }
      />
      {companies.loading && <Spinner />}
      {companies.error && <ErrorBanner message={companies.error} onRetry={companies.reload} />}
      {companies.data && !companies.data.length && <EmptyState title="No companies yet" body="Upload financial documents to analyse a company." />}
      {error && <ErrorBanner message={error} />}
      {loading && !analysis && <Spinner label="Calculating…" />}

      {analysis && (
        <div className={loading ? "opacity-60" : ""}>
          {analysis.notes.map((note) => (
            <p key={note} className="mb-4 rounded-lg border border-line bg-raised px-3 py-2 text-sm text-ink-soft">{note}</p>
          ))}

          {analysis.kpis.length > 0 && (
            <>
              <p className="mb-2 text-xs text-ink-muted">
                {analysis.kpis[0].period}
                {analysis.company.currency ? ` · ${analysis.company.currency} ${analysis.company.unit ?? ""}` : ""} · change versus prior year
              </p>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {analysis.kpis.map((kpi) => <StatTile key={kpi.key} calc={kpi} />)}
              </div>
            </>
          )}

          {analysis.charts.length > 0 && (
            <>
              <div className="mb-3 mt-8 flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-sm font-semibold text-ink">Trends</h2>
                <label className="flex items-center gap-2 text-xs text-ink-soft">
                  <input type="checkbox" checked={projection} onChange={(event) => setProjection(event.target.checked)} />
                  Show illustrative revenue extrapolation
                </label>
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                {analysis.charts.map((chart) => (
                  <div key={chart.id} className="card p-5">
                    <DataChart chart={chart} />
                  </div>
                ))}
              </div>
            </>
          )}

          {analysis.calculations.length > 0 && (
            <Card
              title="Financial ratios"
              subtitle={`${analysis.calculations[0].period} · formulas and inputs shown; page links open the source table`}
              className="mt-6"
              actions={
                <div className="flex flex-wrap gap-1" role="tablist" aria-label="Ratio category">
                  {CATEGORIES.map((name) => (
                    <button
                      key={name}
                      role="tab"
                      aria-selected={category === name}
                      onClick={() => setCategory(name)}
                      className={`rounded-md px-2.5 py-1 text-xs capitalize ${category === name ? "bg-accent-soft font-medium text-accent-ink" : "text-ink-soft hover:bg-page"}`}
                    >
                      {name}
                    </button>
                  ))}
                </div>
              }
            >
              <div className="grid gap-2 md:grid-cols-2">
                {analysis.calculations.filter((c) => c.category === category).map((calc) => (
                  <CalculationCard key={calc.key} calc={{ ...calc, id: "" }} onSource={setPreview} />
                ))}
              </div>
            </Card>
          )}

          {analysis.periods.length > 0 && (
            <Card title="Risk signals" subtitle="Rule-based checks comparing the latest year with the one before" className="mt-6">
              <RiskFlagList flags={analysis.risk_flags} />
            </Card>
          )}

          {analysis.tables.map((table) => (
            <Card key={table.id} className="mt-6">
              <TableView table={table} />
            </Card>
          ))}

          {analysis.periods.length > 0 && (
            <Card
              title="Analyst commentary"
              subtitle="Written from the retrieved passages and the figures above, with citations"
              className="mt-6"
              actions={
                <button onClick={() => void writeNarrative()} disabled={narrative?.loading} className="btn-ghost">
                  <Sparkles className="h-4 w-4" aria-hidden />
                  {narrative ? "Regenerate" : "Generate"}
                </button>
              }
            >
              {!narrative && <p className="text-sm text-ink-soft">Generate a short evidence-backed summary of this company&apos;s performance.</p>}
              {narrative?.loading && <Spinner label="Retrieving evidence and writing…" />}
              {narrative && !narrative.loading && <Markdown content={narrative.text} />}
            </Card>
          )}

          <div className="mt-6">
            <Disclaimer text={analysis.disclaimer} />
          </div>
        </div>
      )}
      <SourcePanel chunkId={preview} onClose={() => setPreview(null)} />
    </>
  );
}

export default function AnalysisPage() {
  return (
    <Suspense>
      <AnalysisView />
    </Suspense>
  );
}
