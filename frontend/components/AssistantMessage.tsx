"use client";

import { ChevronDown, FileText, Loader2, ShieldAlert, ShieldCheck, ShieldQuestion } from "lucide-react";

import { sourceLabel, titleCase } from "@/lib/format";
import type { ChatResponse, Citation } from "@/lib/types";

import { CalculationCard, RiskFlagList, TableView } from "./Artifacts";
import { DataChart } from "./DataChart";
import { Markdown } from "./Markdown";
import { Badge } from "./ui";

export interface AssistantTurn {
  content: string;
  streaming: boolean;
  status?: string;
  error?: string;
  data: Partial<ChatResponse>;
}

const MODE_LABEL: Record<string, string> = {
  generative: "Generated from sources",
  extractive: "Extractive (no LLM)",
  blocked: "Blocked by guardrail",
  insufficient_evidence: "Insufficient evidence",
  general_knowledge: "General knowledge",
};

function ValidationBadge({ data }: { data: Partial<ChatResponse> }) {
  const validation = data.validation;
  if (!validation || validation.status === "not_applicable") return null;
  const config = {
    grounded: { Icon: ShieldCheck, tone: "good" as const, text: "Grounded: every figure and citation checked against the sources" },
    partially_grounded: { Icon: ShieldAlert, tone: "serious" as const, text: "Partially grounded: some figures or citations could not be verified" },
    ungrounded: { Icon: ShieldQuestion, tone: "critical" as const, text: "Ungrounded: the answer cites no source" },
  }[validation.status];
  return (
    <div className="rounded-lg border border-line bg-raised px-3 py-2">
      <p className="flex items-center gap-2 text-xs font-medium text-ink">
        <config.Icon className="h-4 w-4 text-ink-soft" aria-hidden />
        <Badge tone={config.tone}>{titleCase(validation.status)}</Badge>
        <span className="font-normal text-ink-soft">{config.text}</span>
      </p>
      {validation.warnings.length > 0 && (
        <ul className="mt-1.5 list-disc space-y-0.5 pl-9 text-xs text-ink-soft">
          {validation.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function SourceList({ sources, cited, onOpen }: { sources: Citation[]; cited: Set<string>; onOpen: (citation: Citation) => void }) {
  if (!sources.length) return null;
  // Cited sources first; the rest were retrieved but not used in the answer.
  const ordered = [...sources].sort((a, b) => Number(cited.has(b.id)) - Number(cited.has(a.id)));
  return (
    <div>
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-soft">Evidence</p>
      <ul className="grid gap-2 sm:grid-cols-2">
        {ordered.map((source) => (
          <li key={source.id}>
            <button
              onClick={() => onOpen(source)}
              className={`flex w-full items-start gap-2.5 rounded-lg border border-line bg-raised p-2.5 text-left hover:border-accent ${
                cited.size && !cited.has(source.id) ? "opacity-60" : ""
              }`}
            >
              <FileText className="mt-0.5 h-4 w-4 shrink-0 text-ink-muted" aria-hidden />
              <span className="min-w-0">
                <span className="block text-xs font-medium text-ink">
                  <span className="mr-1.5 font-mono text-[10.5px] text-accent-ink">{source.id}</span>
                  {sourceLabel(source)}
                </span>
                <span className="mt-0.5 block truncate text-[11px] text-ink-muted">
                  {[source.company_name, source.section].filter(Boolean).join(" · ")}
                </span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function AssistantMessage({ turn, onOpenSource, onOpenChunk, onFollowUp }: {
  turn: AssistantTurn;
  onOpenSource: (citation: Citation) => void;
  onOpenChunk: (chunkId: string) => void;
  onFollowUp: (question: string) => void;
}) {
  const { data } = turn;
  const sources = data.sources || [];
  const calculations = data.calculations || [];
  const metrics = calculations.filter((c) => c.kind !== "flag");
  const flags = calculations.filter((c) => c.kind === "flag");
  const cited = new Set((data.citations || []).map((c) => c.id));
  // Extractive answers already list the data notes in their text.
  const notes = (data.notes || []).filter((note) => !turn.content.includes(note));

  const handleCite = (id: string) => {
    if (id.startsWith("S")) {
      const source = sources.find((s) => s.id === id);
      if (source) onOpenSource(source);
      return;
    }
    const element = document.getElementById(`artifact-${id}`);
    element?.closest("details")?.setAttribute("open", "");
    element?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  return (
    <div className="space-y-4">
      {turn.streaming && turn.status && (
        <p className="flex items-center gap-2 text-xs text-ink-soft" role="status">
          <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
          {turn.status}
        </p>
      )}

      {turn.content && <Markdown content={turn.content} onCite={handleCite} />}
      {turn.error && (
        <p className="rounded-lg border border-line bg-raised px-3 py-2 text-sm text-ink" role="alert">
          {turn.error}
        </p>
      )}

      {(data.charts || []).length > 0 && (
        <div className={`grid gap-4 ${data.charts!.length > 1 ? "md:grid-cols-2" : ""}`}>
          {data.charts!.map((chart) => (
            <div key={chart.id} className="card p-4">
              <DataChart chart={chart} height={data.charts!.length > 1 ? 200 : 240} />
            </div>
          ))}
        </div>
      )}

      {(metrics.length > 0 || flags.length > 0 || (data.tables || []).length > 0) && (
        <details className="group rounded-lg border border-line bg-surface" open={data.mode !== "extractive"}>
          <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-ink-soft">
            <ChevronDown className="h-3.5 w-3.5 transition-transform group-open:rotate-180" aria-hidden />
            Computed in Python ({metrics.length + flags.length + (data.tables || []).length})
          </summary>
          <div className="space-y-4 border-t border-line p-3">
            {metrics.length > 0 && (
              <div className="grid gap-2 md:grid-cols-2">
                {metrics.map((calc) => (
                  <CalculationCard key={calc.id} calc={calc} onSource={onOpenChunk} />
                ))}
              </div>
            )}
            {flags.length > 0 && <RiskFlagList flags={flags} />}
            {(data.tables || []).map((table) => (
              <TableView key={table.id} table={table} />
            ))}
          </div>
        </details>
      )}

      {!turn.streaming && <ValidationBadge data={data} />}
      <SourceList sources={sources} cited={cited} onOpen={onOpenSource} />

      {notes.length > 0 && (
        <ul className="list-disc space-y-0.5 pl-5 text-xs text-ink-soft">
          {notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}

      {!turn.streaming && (data.followups || []).length > 0 && (
        <div className="flex flex-wrap gap-2">
          {data.followups!.map((question) => (
            <button key={question} onClick={() => onFollowUp(question)} className="rounded-full border border-line bg-raised px-3 py-1 text-xs text-ink-soft hover:border-accent hover:text-ink">
              {question}
            </button>
          ))}
        </div>
      )}

      {!turn.streaming && data.intent && (
        <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-ink-muted">
          <span>{titleCase(data.intent)}</span>
          {data.mode && <span>{MODE_LABEL[data.mode] || data.mode}</span>}
          {data.trace?.plan && data.trace.plan.length > 0 && <span>Agents: {data.trace.plan.join(" → ")}</span>}
          {data.trace?.total_ms !== undefined && <span>{(data.trace.total_ms / 1000).toFixed(1)} s</span>}
          {data.trace?.usage?.total_tokens ? <span>{data.trace.usage.total_tokens.toLocaleString()} tokens</span> : null}
        </p>
      )}
    </div>
  );
}
