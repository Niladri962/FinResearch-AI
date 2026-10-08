"use client";

import { Table2 } from "lucide-react";
import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { compactNumber, fullNumber } from "@/lib/format";
import type { ChartSpec } from "@/lib/types";

// Categorical slots are assigned in this fixed order and never cycled.
const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)", "var(--series-6)"];

interface TooltipEntry {
  name?: string;
  value?: number | null;
  color?: string;
}

function ChartTooltip({ active, label, payload, unit }: { active?: boolean; label?: string; payload?: TooltipEntry[]; unit: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-line bg-raised px-3 py-2 text-xs shadow-card">
      <p className="mb-1 font-medium text-ink">{label}</p>
      {payload.map((entry) => (
        <p key={entry.name} className="flex items-center gap-2 text-ink-soft">
          <span className="h-2 w-2 rounded-full" style={{ background: entry.color }} aria-hidden />
          <span>{entry.name}</span>
          <span className="ml-auto pl-3 font-medium tabular-nums text-ink">{fullNumber(entry.value, unit)}</span>
        </p>
      ))}
    </div>
  );
}

export function DataChart({ chart, height = 240 }: { chart: ChartSpec; height?: number }) {
  const [showTable, setShowTable] = useState(false);
  const series = chart.series.slice(0, SERIES.length);
  const rows = chart.x.map((label, index) => {
    const row: Record<string, string | number | null> = { label };
    series.forEach((s) => (row[s.name] = s.data[index] ?? null));
    return row;
  });
  const values = series.flatMap((s) => s.data).filter((v): v is number => v !== null);
  const hasNegative = values.some((v) => v < 0);
  const tick = (value: number) => (chart.unit === "%" ? `${compactNumber(value)}%` : chart.unit === "x" ? `${value}x` : compactNumber(value));
  const axis = { fontSize: 11, fill: "var(--ink-muted)" };

  // An array, not a fragment: Recharts only recognises axes and grids that are direct children.
  const common = [
    <CartesianGrid key="grid" stroke="var(--line)" vertical={false} />,
    <XAxis key="x" dataKey="label" tick={axis} tickLine={false} axisLine={{ stroke: "var(--axis)" }} interval={rows.length <= 8 ? 0 : "preserveStartEnd"} />,
    <YAxis key="y" tick={axis} tickLine={false} axisLine={false} width={52} tickFormatter={tick} />,
    hasNegative ? <ReferenceLine key="zero" y={0} stroke="var(--axis)" /> : null,
    <Tooltip
      key="tooltip"
      content={<ChartTooltip unit={chart.unit} />}
      cursor={chart.kind === "bar" ? { fill: "var(--line)", opacity: 0.35 } : { stroke: "var(--axis)" }}
    />,
  ];

  return (
    <figure className="min-w-0">
      <figcaption className="mb-2 flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-ink">{chart.title}</p>
          {chart.unit && !["%", "x"].includes(chart.unit) && <p className="text-xs text-ink-muted">{chart.unit}</p>}
        </div>
        <button
          onClick={() => setShowTable((v) => !v)}
          className="no-print flex items-center gap-1 rounded-md px-1.5 py-1 text-xs text-ink-soft hover:bg-page"
          aria-pressed={showTable}
        >
          <Table2 className="h-3.5 w-3.5" aria-hidden />
          {showTable ? "Chart" : "Data"}
        </button>
      </figcaption>

      {/* One colour needs no legend box; two or more always get one. */}
      {series.length > 1 && (
        <ul className="mb-2 flex flex-wrap gap-x-4 gap-y-1">
          {series.map((s, index) => (
            <li key={s.name} className="flex items-center gap-1.5 text-xs text-ink-soft">
              <span className="h-2 w-2 rounded-full" style={{ background: SERIES[index] }} aria-hidden />
              {s.name}
            </li>
          ))}
        </ul>
      )}

      {showTable ? (
        <div className="overflow-x-auto rounded-lg border border-line">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="bg-page text-ink-soft">
                <th className="px-3 py-2 font-medium">Period</th>
                {series.map((s) => (
                  <th key={s.name} className="px-3 py-2 text-right font-medium">{s.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={String(row.label)} className="border-t border-line">
                  <td className="px-3 py-1.5">{row.label}</td>
                  {series.map((s) => (
                    <td key={s.name} className="px-3 py-1.5 text-right tabular-nums">
                      {fullNumber(row[s.name] as number | null, chart.unit)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ height }} role="img" aria-label={`${chart.title}. Use the Data button for a table of values.`}>
          <ResponsiveContainer width="100%" height="100%">
            {chart.kind === "bar" ? (
              <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barGap={2} barCategoryGap="28%">
                {common}
                {series.map((s, index) => (
                  <Bar key={s.name} dataKey={s.name} fill={SERIES[index]} maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false} />
                ))}
              </BarChart>
            ) : (
              <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                {common}
                {series.map((s, index) => (
                  <Line
                    key={s.name}
                    dataKey={s.name}
                    type="linear"
                    stroke={SERIES[index]}
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    dot={{ r: 4, fill: SERIES[index], stroke: "var(--surface)", strokeWidth: 2 }}
                    activeDot={{ r: 5, stroke: "var(--surface)", strokeWidth: 2 }}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            )}
          </ResponsiveContainer>
        </div>
      )}
    </figure>
  );
}
