"use client";

import { AlertOctagon, AlertTriangle, ArrowDownRight, ArrowUpRight, Calculator, Info } from "lucide-react";

import { signed } from "@/lib/format";
import type { Calculation, DataTable } from "@/lib/types";

import { Badge } from "./ui";

/** A headline figure with its change versus the prior period. */
export function StatTile({ calc }: { calc: Calculation }) {
  const change = calc.change;
  return (
    <div className="card px-4 py-3.5">
      <p className="text-xs text-ink-soft">{calc.name}</p>
      <p className="mt-1 text-xl font-semibold tracking-tight text-ink">{calc.display}</p>
      <p className="mt-1 flex items-center gap-1 text-xs text-ink-muted">
        {change !== null && change !== undefined ? (
          <>
            {change >= 0 ? <ArrowUpRight className="h-3.5 w-3.5" aria-hidden /> : <ArrowDownRight className="h-3.5 w-3.5" aria-hidden />}
            <span className="font-medium text-ink-soft">{signed(change, calc.change_unit)}</span>
            <span>vs prior year</span>
          </>
        ) : (
          <span>{calc.period || "—"}</span>
        )}
      </p>
    </div>
  );
}

/** A computed metric with formula and traceable inputs. */
export function CalculationCard({ calc, onSource }: { calc: Calculation; onSource?: (chunkId: string) => void }) {
  return (
    <div id={`artifact-${calc.id}`} className="rounded-lg border border-line bg-raised p-3 scroll-mt-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="flex items-center gap-1.5 text-xs text-ink-soft">
            <Calculator className="h-3.5 w-3.5 shrink-0" aria-hidden />
            <span className="truncate">
              {calc.id && <span className="mr-1 font-mono text-[10.5px] text-accent-ink">{calc.id}</span>}
              {[calc.company, calc.period].filter(Boolean).join(" · ")}
            </span>
          </p>
          <p className="mt-0.5 text-sm font-medium text-ink">{calc.name}</p>
        </div>
        <p className="shrink-0 text-lg font-semibold tabular-nums text-ink">{calc.display}</p>
      </div>
      {calc.kind !== "metric" && calc.formula && <p className="mt-2 font-mono text-[11px] text-ink-soft">{calc.formula}</p>}
      {calc.inputs.length > 0 && calc.kind !== "metric" && (
        <dl className="mt-2 space-y-1 border-t border-line pt-2">
          {calc.inputs.map((input) => {
            const source = input.sources[0];
            return (
              <div key={input.key} className="flex items-baseline justify-between gap-3 text-xs">
                <dt className="text-ink-soft">
                  {input.name}
                  {input.derived && <span className="ml-1 text-ink-muted">(derived)</span>}
                </dt>
                <dd className="flex items-baseline gap-2 text-right">
                  <span className="tabular-nums text-ink">{input.display}</span>
                  {source?.chunk_id && onSource && (
                    <button onClick={() => onSource(source.chunk_id as string)} className="text-accent hover:underline">
                      p.{source.page}
                    </button>
                  )}
                </dd>
              </div>
            );
          })}
        </dl>
      )}
      {calc.note && <p className="mt-2 text-[11px] leading-relaxed text-ink-muted">{calc.note}</p>}
    </div>
  );
}

const SEVERITY = {
  high: { tone: "critical" as const, label: "High", Icon: AlertOctagon },
  medium: { tone: "serious" as const, label: "Medium", Icon: AlertTriangle },
  low: { tone: "warn" as const, label: "Low", Icon: Info },
  info: { tone: "neutral" as const, label: "Info", Icon: Info },
};

export function RiskFlagList({ flags }: { flags: Calculation[] }) {
  if (!flags.length) return <p className="text-sm text-ink-soft">No quantitative risk signals were triggered by the latest reported figures.</p>;
  return (
    <ul className="space-y-2.5">
      {flags.map((flag) => {
        const severity = SEVERITY[flag.severity || "info"];
        return (
          <li key={flag.key + flag.company} id={`artifact-${flag.id}`} className="flex items-start gap-3 scroll-mt-4">
            <severity.Icon className="mt-0.5 h-4 w-4 shrink-0 text-ink-soft" aria-hidden />
            <div className="min-w-0 flex-1">
              <p className="flex flex-wrap items-center gap-2 text-sm font-medium text-ink">
                {flag.name}
                <Badge tone={severity.tone}>{severity.label}</Badge>
              </p>
              <p className="mt-0.5 text-sm text-ink-soft">{flag.display}</p>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

export function TableView({ table }: { table: DataTable }) {
  return (
    <div id={`artifact-${table.id}`} className="scroll-mt-4">
      <p className="mb-2 text-sm font-medium text-ink">{table.title}</p>
      <div className="overflow-x-auto rounded-lg border border-line">
        <table className="w-full text-left text-[13px]">
          <thead>
            <tr className="bg-page text-ink-soft">
              {table.columns.map((column, index) => (
                <th key={column} className={`px-3 py-2 font-medium ${index ? "text-right" : ""}`}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row) => (
              <tr key={row[0]} className="border-t border-line">
                {row.map((cell, index) => (
                  <td
                    key={index}
                    className={`px-3 py-1.5 ${index ? "text-right tabular-nums" : "text-ink"} ${cell === "Not available" ? "text-ink-muted" : ""}`}
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.note && <p className="mt-2 text-xs leading-relaxed text-ink-muted">{table.note}</p>}
    </div>
  );
}
