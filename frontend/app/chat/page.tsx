"use client";

import { MessageSquarePlus, SendHorizontal, Square, Trash2 } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useRef, useState } from "react";

import { AssistantMessage, type AssistantTurn } from "@/components/AssistantMessage";
import { SourcePanel } from "@/components/SourcePanel";
import { Disclaimer, ErrorBanner } from "@/components/ui";
import { api, ApiError, streamChat } from "@/lib/api";
import { relativeTime } from "@/lib/format";
import type { Citation, Company, ConversationSummary } from "@/lib/types";

type Turn = { id: string; role: "user"; content: string } | ({ id: string; role: "assistant" } & AssistantTurn);

const EXAMPLES = [
  "Why did operating margins decline?",
  "What is the company's debt-to-equity ratio?",
  "What are the company's biggest risks?",
  "How has revenue changed over the last 5 years?",
  "Summarize the latest earnings call.",
  "What does management expect for next year?",
];

let counter = 0;
const nextId = () => `local-${Date.now()}-${counter++}`;

function ChatView() {
  const params = useSearchParams();
  const [companies, setCompanies] = useState<Company[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [companyIds, setCompanyIds] = useState<number[]>([]);
  const [years, setYears] = useState<number[]>([]);
  const [preview, setPreview] = useState<{ chunkId: string; label?: string } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const autoAsked = useRef(false);

  const refreshConversations = useCallback(() => {
    api.conversations().then(setConversations).catch(() => undefined);
  }, []);

  useEffect(() => {
    api.companies().then(setCompanies).catch((err) => setError(err instanceof ApiError ? err.message : "Could not load companies."));
    refreshConversations();
  }, [refreshConversations]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [turns]);

  const patchLast = (patch: (turn: AssistantTurn) => Partial<AssistantTurn>) =>
    setTurns((current) => {
      const last = current[current.length - 1];
      if (!last || last.role !== "assistant") return current;
      return [...current.slice(0, -1), { ...last, ...patch(last) }];
    });

  const send = useCallback(
    async (text: string, overrideConversation?: string | null) => {
      const message = text.trim();
      if (!message || busy) return;
      setInput("");
      setError(null);
      setBusy(true);
      setTurns((current) => [
        ...current,
        { id: nextId(), role: "user", content: message },
        { id: nextId(), role: "assistant", content: "", streaming: true, status: "Understanding the question…", data: {} },
      ]);
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const stream = streamChat(
          {
            message,
            conversation_id: overrideConversation === undefined ? conversationId : overrideConversation,
            filters: { company_ids: companyIds, document_ids: [], fiscal_years: years, document_types: [] },
          },
          controller.signal,
        );
        for await (const item of stream) {
          switch (item.event) {
            case "conversation":
              setConversationId(item.data.conversation_id);
              break;
            case "status":
              patchLast(() => ({ status: item.data.message }));
              break;
            case "meta":
              patchLast((turn) => ({ data: { ...turn.data, intent: item.data.intent } }));
              break;
            case "artifacts":
              patchLast((turn) => ({ data: { ...turn.data, ...item.data } }));
              break;
            case "sources":
              patchLast((turn) => ({ data: { ...turn.data, sources: item.data.sources } }));
              break;
            case "token":
              patchLast((turn) => ({ content: turn.content + item.data.text }));
              break;
            case "reset":
              patchLast(() => ({ content: "" }));
              break;
            case "final":
              // The verified answer replaces the streamed draft.
              patchLast(() => ({ content: item.data.answer, streaming: false, status: undefined, data: item.data }));
              break;
            case "error":
              patchLast(() => ({ streaming: false, status: undefined, error: item.data.message }));
              break;
          }
        }
      } catch (err) {
        patchLast(() => ({ error: err instanceof ApiError ? err.message : "The connection was interrupted." }));
      } finally {
        patchLast((turn) => (turn.streaming ? { streaming: false, status: undefined, error: turn.error || (turn.content ? undefined : "Stopped.") } : {}));
        setBusy(false);
        abortRef.current = null;
        refreshConversations();
      }
    },
    [busy, companyIds, conversationId, refreshConversations, years],
  );

  const openConversation = useCallback(async (id: string) => {
    try {
      const detail = await api.conversation(id);
      setConversationId(id);
      setTurns(
        detail.messages.map((m) =>
          m.role === "user"
            ? { id: m.id, role: "user", content: m.content }
            : { id: m.id, role: "assistant", content: m.content, streaming: false, data: m.payload },
        ),
      );
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load the conversation.");
    }
  }, []);

  // Deep links: /chat?c=<id> opens a saved thread, /chat?q=… asks a question once.
  useEffect(() => {
    if (autoAsked.current) return;
    const conversation = params.get("c");
    const question = params.get("q");
    if (conversation) {
      autoAsked.current = true;
      void openConversation(conversation);
    } else if (question) {
      autoAsked.current = true;
      void send(question, null);
    }
  }, [params, send, openConversation]);

  const newConversation = () => {
    abortRef.current?.abort();
    setConversationId(null);
    setTurns([]);
    setError(null);
  };

  const removeConversation = async (id: string) => {
    await api.deleteConversation(id).catch(() => undefined);
    if (id === conversationId) newConversation();
    refreshConversations();
  };

  const toggle = <T,>(list: T[], value: T) => (list.includes(value) ? list.filter((v) => v !== value) : [...list, value]);
  const scoped = companyIds.length ? companies.filter((c) => companyIds.includes(c.id)) : companies;
  const availableYears = [...new Set(scoped.flatMap((c) => c.fiscal_years))].sort((a, b) => b - a);
  const chip = (active: boolean) =>
    `rounded-full border px-2.5 py-1 text-xs transition-colors ${
      active ? "border-accent bg-accent-soft font-medium text-accent-ink" : "border-line bg-raised text-ink-soft hover:border-accent"
    }`;

  return (
    <div className="flex h-full">
      {/* Conversation history */}
      <aside className="hidden w-64 shrink-0 flex-col border-r border-line bg-surface md:flex">
        <div className="p-3">
          <button onClick={newConversation} className="btn-ghost w-full">
            <MessageSquarePlus className="h-4 w-4" aria-hidden />
            New research thread
          </button>
        </div>
        <ul className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-3">
          {conversations.map((conversation) => (
            <li key={conversation.id} className="group relative">
              <button
                onClick={() => !busy && void openConversation(conversation.id)}
                className={`w-full rounded-lg px-3 py-2 pr-8 text-left ${conversation.id === conversationId ? "bg-accent-soft" : "hover:bg-page"}`}
              >
                <span className="block truncate text-sm text-ink">{conversation.title}</span>
                <span className="text-[11px] text-ink-muted">{relativeTime(conversation.updated_at)}</span>
              </button>
              <button
                onClick={() => void removeConversation(conversation.id)}
                aria-label={`Delete conversation: ${conversation.title}`}
                className="absolute right-1.5 top-2.5 rounded p-1 text-ink-muted opacity-0 hover:text-critical focus:opacity-100 group-hover:opacity-100"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </li>
          ))}
          {!conversations.length && <li className="px-3 py-2 text-xs text-ink-muted">No conversations yet.</li>}
        </ul>
      </aside>

      {/* Thread */}
      <section className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl px-5 py-6">
            {error && <ErrorBanner message={error} />}
            {!turns.length ? (
              <div className="pt-10">
                <h1 className="text-xl font-semibold tracking-tight">Research Chat</h1>
                <p className="mt-1 text-sm text-ink-soft">
                  Ask about your uploaded filings. Answers cite the exact page they come from, and every ratio is calculated in code, not by the model.
                </p>
                {!companies.length && (
                  <p className="mt-4 rounded-lg border border-line bg-raised px-3 py-2 text-sm text-ink-soft">
                    No documents are indexed yet. Upload filings on the Documents page to get evidence-backed answers.
                  </p>
                )}
                <div className="mt-6 grid gap-2 sm:grid-cols-2">
                  {EXAMPLES.map((example) => (
                    <button key={example} onClick={() => void send(example)} className="card px-4 py-3 text-left text-sm text-ink hover:border-accent">
                      {example}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <ol className="space-y-7">
                {turns.map((turn) =>
                  turn.role === "user" ? (
                    <li key={turn.id} className="flex justify-end">
                      <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-sm text-white">{turn.content}</p>
                    </li>
                  ) : (
                    <li key={turn.id}>
                      <AssistantMessage
                        turn={turn}
                        onOpenSource={(citation: Citation) => setPreview({ chunkId: citation.chunk_id, label: citation.id })}
                        onOpenChunk={(chunkId) => setPreview({ chunkId })}
                        onFollowUp={(question) => void send(question)}
                      />
                    </li>
                  ),
                )}
              </ol>
            )}
            <div ref={bottomRef} className="h-4" />
          </div>
        </div>

        {/* Composer */}
        <div className="border-t border-line bg-surface">
          <div className="mx-auto max-w-3xl px-5 py-3">
            {companies.length > 0 && (
              <div className="mb-2 flex flex-wrap items-center gap-1.5">
                <span className="mr-1 text-[11px] font-medium uppercase tracking-wide text-ink-muted">Scope</span>
                {companies.map((company) => (
                  <button key={company.id} onClick={() => setCompanyIds((ids) => toggle(ids, company.id))} aria-pressed={companyIds.includes(company.id)} className={chip(companyIds.includes(company.id))}>
                    {company.name}
                  </button>
                ))}
                {availableYears.map((year) => (
                  <button key={year} onClick={() => setYears((list) => toggle(list, year))} aria-pressed={years.includes(year)} className={chip(years.includes(year))}>
                    FY{year}
                  </button>
                ))}
              </div>
            )}
            <form
              onSubmit={(event) => {
                event.preventDefault();
                void send(input);
              }}
              className="flex items-end gap-2"
            >
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void send(input);
                  }
                }}
                rows={1}
                maxLength={2000}
                placeholder="Ask a question about your documents…"
                aria-label="Question"
                className="input max-h-40 min-h-[42px] flex-1 resize-y"
              />
              {busy ? (
                <button type="button" onClick={() => abortRef.current?.abort()} className="btn-ghost h-[42px]" aria-label="Stop generating">
                  <Square className="h-4 w-4" aria-hidden />
                  Stop
                </button>
              ) : (
                <button type="submit" disabled={!input.trim()} className="btn-primary h-[42px]">
                  <SendHorizontal className="h-4 w-4" aria-hidden />
                  Ask
                </button>
              )}
            </form>
            <div className="mt-2">
              <Disclaimer text="Research assistant, not a financial advisor. Answers are generated from your documents and may contain errors — verify against the cited pages." />
            </div>
          </div>
        </div>
      </section>

      <SourcePanel chunkId={preview?.chunkId ?? null} label={preview?.label} onClose={() => setPreview(null)} />
    </div>
  );
}

export default function ChatPage() {
  return (
    <Suspense>
      <ChatView />
    </Suspense>
  );
}
