"use client";

import { useEffect, useState } from "react";

import { Badge, Card, Disclaimer, ErrorBanner, PageHeader, Spinner } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { api, API_URL, getApiKey, setApiKey } from "@/lib/api";
import { relativeTime, titleCase } from "@/lib/format";

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5 text-sm">
      <dt className="text-ink-soft">{label}</dt>
      <dd className="text-right font-medium text-ink">{value}</dd>
    </div>
  );
}

export default function SettingsPage() {
  const system = useApi(api.system);
  const health = useApi(api.health);
  const traces = useApi(() => api.traces(15));
  const [key, setKey] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => setKey(getApiKey()), []);

  const save = () => {
    setApiKey(key.trim());
    setSaved(true);
    system.reload();
    traces.reload();
  };

  const info = system.data;
  const tone = (status: string) => (status === "ok" ? "good" : status === "not_configured" ? "neutral" : status === "degraded" ? "warn" : "critical");

  return (
    <>
      <PageHeader title="Settings" subtitle="Runtime configuration is read-only here and is changed through environment variables on the server." />

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Connection" subtitle={`API: ${API_URL}`}>
          {health.loading && <Spinner />}
          {health.error && <ErrorBanner message={health.error} onRetry={health.reload} />}
          {health.data && (
            <ul className="space-y-2">
              {Object.entries(health.data.components).map(([name, component]) => (
                <li key={name} className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-ink">{titleCase(name)}</span>
                  <span className="flex items-center gap-2 text-xs text-ink-muted">
                    {String(component.backend ?? component.provider ?? "")}
                    <Badge tone={tone(component.status)}>{titleCase(component.status)}</Badge>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card title="Access key" subtitle="Only needed when the server runs with AUTH_ENABLED=true">
          <label className="block">
            <span className="label">API access key</span>
            <input
              type="password"
              className="input"
              autoComplete="off"
              value={key}
              onChange={(event) => {
                setKey(event.target.value);
                setSaved(false);
              }}
              placeholder={info?.auth_enabled ? "Required" : "Not required on this server"}
            />
          </label>
          <div className="mt-3 flex items-center gap-3">
            <button onClick={save} className="btn-primary">Save</button>
            {saved && <span className="text-xs text-ink-soft">Saved in this browser.</span>}
          </div>
          <p className="mt-3 text-xs leading-relaxed text-ink-muted">
            This key identifies you to the FinResearch API and is stored only in this browser. Model-provider keys (LLM, embeddings, Qdrant) are configured on the server and are never sent to the browser.
          </p>
        </Card>

        {system.error && <ErrorBanner message={system.error} onRetry={system.reload} />}
        {info && (
          <>
            <Card title="Models">
              <dl className="divide-y divide-line">
                <Row label="Answer mode" value={<Badge tone={info.llm.configured ? "good" : "neutral"}>{info.llm.configured ? "Generative" : "Extractive (no LLM configured)"}</Badge>} />
                <Row label="LLM provider" value={info.llm.provider} />
                <Row label="LLM model" value={info.llm.model || "—"} />
                <Row label="Embeddings" value={`${info.embeddings.model} (${info.embeddings.provider})`} />
                <Row label="Reranker" value={info.reranker.model} />
              </dl>
            </Card>

            <Card title="Retrieval">
              <dl className="divide-y divide-line">
                <Row label="Semantic candidates" value={info.retrieval.semantic_top_k} />
                <Row label="Keyword (BM25) candidates" value={info.retrieval.bm25_top_k} />
                <Row label="Fusion" value={`${info.retrieval.fusion_method} · semantic ${info.retrieval.semantic_weight} / keyword ${info.retrieval.keyword_weight}`} />
                <Row label="Contexts after reranking" value={info.retrieval.rerank_top_n} />
                <Row label="Chunk size / overlap" value={`${info.retrieval.chunk_target_tokens} / ${info.retrieval.chunk_overlap_tokens} tokens`} />
              </dl>
            </Card>

            <Card title="Infrastructure">
              <dl className="divide-y divide-line">
                <Row label="Environment" value={info.environment} />
                <Row label="Version" value={info.version} />
                <Row label="Vector store" value={info.vector_store.backend} />
                <Row label="Database" value={info.database} />
                <Row label="Cache" value={info.cache} />
                <Row label="Authentication" value={info.auth_enabled ? "API key required" : "Disabled"} />
              </dl>
            </Card>

            <Card title="Limits">
              <dl className="divide-y divide-line">
                <Row label="Max upload size" value={`${info.limits.max_upload_mb} MB`} />
                <Row label="Accepted formats" value={info.limits.allowed_extensions.join(" ")} />
                <Row label="Rate limit" value={info.limits.rate_limit_per_minute ? `${info.limits.rate_limit_per_minute} requests / minute` : "Disabled"} />
                <Row label="Max question length" value={`${info.limits.max_query_chars} characters`} />
              </dl>
            </Card>
          </>
        )}
      </div>

      <Card title="Recent query traces" subtitle="Latency per stage and token usage. Query text is stored as a hash unless TRACE_QUERY_TEXT is enabled." className="mt-4">
        {traces.loading && <Spinner />}
        {traces.error && <p className="text-sm text-ink-soft">{traces.error}</p>}
        {traces.data && !traces.data.length && <p className="text-sm text-ink-soft">No queries yet.</p>}
        {traces.data && traces.data.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-line text-ink-soft">
                  {["When", "Intent", "Mode", "Chunks", "Retrieval", "Rerank", "LLM", "Total", "Tokens", "Grounding"].map((h) => (
                    <th key={h} className="whitespace-nowrap px-2 py-2 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="tabular-nums">
                {traces.data.map((trace) => (
                  <tr key={trace.id} className="border-b border-line last:border-b-0">
                    <td className="whitespace-nowrap px-2 py-1.5 text-ink-muted">{relativeTime(trace.created_at)}</td>
                    <td className="whitespace-nowrap px-2 py-1.5">{trace.intent ? titleCase(trace.intent) : "—"}</td>
                    <td className="whitespace-nowrap px-2 py-1.5">{trace.status === "ok" ? titleCase(trace.mode || "—") : titleCase(trace.status)}</td>
                    <td className="px-2 py-1.5">{trace.retrieved_chunks ?? "—"}</td>
                    <td className="px-2 py-1.5">{trace.timings_ms?.retrieval != null ? `${trace.timings_ms.retrieval.toFixed(0)} ms` : "—"}</td>
                    <td className="px-2 py-1.5">{trace.timings_ms?.rerank != null ? `${trace.timings_ms.rerank.toFixed(0)} ms` : "—"}</td>
                    <td className="px-2 py-1.5">{trace.timings_ms?.llm != null ? `${trace.timings_ms.llm.toFixed(0)} ms` : "—"}</td>
                    <td className="px-2 py-1.5">{(trace.total_ms / 1000).toFixed(2)} s</td>
                    <td className="px-2 py-1.5">{trace.usage?.total_tokens || "—"}</td>
                    <td className="whitespace-nowrap px-2 py-1.5">{trace.validation_status ? titleCase(trace.validation_status) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <div className="mt-6">
        <Disclaimer text={info?.limits.disclaimer} />
      </div>
    </>
  );
}
