import type {
  AnalyzeResponse,
  ApiErrorBody,
  ChatEvent,
  ChatResponse,
  ChunkPreview,
  Company,
  CompareResponse,
  ConversationDetail,
  ConversationSummary,
  DashboardData,
  DocumentItem,
  HealthData,
  QueryFilters,
  ReportDetail,
  ReportSummary,
  SystemInfo,
  TraceRow,
  UploadResponse,
} from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
const KEY_STORAGE = "finresearch.apiKey";

/** Why this deployment cannot reach its backend, or null when the configuration looks sound.
 *  Catches the two mistakes that otherwise fail silently on a hosted frontend. */
export function apiConfigProblem(): string | null {
  if (typeof window === "undefined") return null;
  const pageIsLocal = ["localhost", "127.0.0.1"].includes(window.location.hostname);
  if (pageIsLocal) return null;
  if (/\/\/(localhost|127\.0\.0\.1)(:|\/|$)/.test(API_URL)) {
    return "This deployment has no backend configured. Set NEXT_PUBLIC_API_URL to your API's public URL in the hosting dashboard and redeploy.";
  }
  if (window.location.protocol === "https:" && API_URL.startsWith("http://")) {
    return "The backend URL uses http:// but this site is served over https://, so the browser blocks the requests. Use an https:// backend URL in NEXT_PUBLIC_API_URL and redeploy.";
  }
  return null;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

/** Access key for deployments with AUTH_ENABLED=true. It identifies the user to
 *  this app's API only; model-provider keys never leave the server. */
export function getApiKey(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(KEY_STORAGE) || "";
}

export function setApiKey(key: string): void {
  if (key) window.localStorage.setItem(KEY_STORAGE, key);
  else window.localStorage.removeItem(KEY_STORAGE);
}

function authHeaders(): Record<string, string> {
  const key = getApiKey();
  return key ? { "X-API-Key": key } : {};
}

async function toError(response: Response): Promise<ApiError> {
  let body: { error?: ApiErrorBody } = {};
  try {
    body = await response.json();
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(
    response.status,
    body.error?.code || "http_error",
    body.error?.message || `Request failed (HTTP ${response.status}).`,
  );
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/api${path}`, {
      ...init,
      headers: {
        ...(init.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
        ...authHeaders(),
        ...init.headers,
      },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(0, "network_error", apiConfigProblem() || `Cannot reach the API at ${API_URL}. Is the backend running?`);
  }
  if (!response.ok) throw await toError(response);
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

const post = <T>(path: string, body: unknown) => request<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => request<HealthData>("/health"),
  system: () => request<SystemInfo>("/system"),
  dashboard: () => request<DashboardData>("/dashboard"),
  traces: (limit = 25) => request<TraceRow[]>(`/observability/traces?limit=${limit}`),

  companies: () => request<Company[]>("/companies"),
  documents: () => request<DocumentItem[]>("/documents"),
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" }),
  chunk: (chunkId: string) => request<ChunkPreview>(`/documents/chunks/${chunkId}`),
  upload: (files: File[], meta: { company?: string; fiscal_year?: string; quarter?: string; document_type?: string }) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    Object.entries(meta).forEach(([key, value]) => value && form.append(key, value));
    return request<UploadResponse>("/documents/upload", { method: "POST", body: form });
  },

  conversations: () => request<ConversationSummary[]>("/chat/conversations"),
  conversation: (id: string) => request<ConversationDetail>(`/chat/conversations/${id}`),
  deleteConversation: (id: string) => request<void>(`/chat/conversations/${id}`, { method: "DELETE" }),

  analyze: (body: {
    company_id: number;
    analysis_type?: "overview" | "trend" | "risk";
    metrics?: string[];
    include_projection?: boolean;
    include_narrative?: boolean;
  }) => post<AnalyzeResponse>("/analyze", body),
  compare: (body: { company_ids: number[]; periods?: string[]; include_narrative?: boolean }) =>
    post<CompareResponse>("/compare", body),

  reports: () => request<ReportSummary[]>("/reports"),
  report: (id: string) => request<ReportDetail>(`/reports/${id}`),
  generateReport: (companyId: number) => post<ReportDetail>("/reports/generate", { company_id: companyId }),
  deleteReport: (id: string) => request<void>(`/reports/${id}`, { method: "DELETE" }),
};

/** POST /api/chat and yield server-sent events as they arrive. */
export async function* streamChat(
  body: { message: string; conversation_id?: string | null; filters: QueryFilters },
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream", ...authHeaders() },
      body: JSON.stringify({ ...body, stream: true }),
      signal,
    });
  } catch (error) {
    if ((error as Error).name === "AbortError") return;
    throw new ApiError(0, "network_error", `Cannot reach the API at ${API_URL}. Is the backend running?`);
  }
  if (!response.ok || !response.body) throw await toError(response);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        let event = "message";
        let data = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (data) yield { event, data: JSON.parse(data) } as ChatEvent;
        boundary = buffer.indexOf("\n\n");
      }
    }
  } catch (error) {
    if ((error as Error).name !== "AbortError") throw error;
  }
}

export type { ChatResponse };
