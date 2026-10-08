// Mirrors backend/app/models/schemas.py

export interface Company {
  id: number;
  name: string;
  document_count: number;
  fiscal_years: number[];
  periods_with_data: string[];
  currency: string | null;
  unit: string | null;
}

export type DocumentStatus = "queued" | "processing" | "ready" | "failed";

export interface DocumentItem {
  id: string;
  filename: string;
  title: string;
  company_id: number | null;
  company_name: string | null;
  document_type: string;
  document_type_label: string;
  fiscal_year: number | null;
  quarter: number | null;
  reporting_period: string | null;
  currency: string | null;
  unit: string | null;
  status: DocumentStatus;
  progress?: string | null;
  error: string | null;
  warnings: string[];
  page_count: number;
  chunk_count: number;
  fact_count: number;
  size_bytes: number;
  uploaded_at: string;
  processed_at: string | null;
}

export interface ApiErrorBody {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface UploadResult {
  filename: string;
  document: DocumentItem | null;
  error: ApiErrorBody | null;
}

export interface UploadResponse {
  results: UploadResult[];
  accepted: number;
  rejected: number;
}

export interface Citation {
  id: string;
  chunk_id: string;
  document_id: string;
  document_title: string;
  filename: string;
  company_name: string | null;
  page: number;
  page_end: number | null;
  section: string;
  chunk_type: string;
  snippet: string;
  score: number | null;
}

export interface ChunkPreview {
  id: string;
  document_id: string;
  document_title: string;
  company_name: string | null;
  text: string;
  page_start: number;
  page_end: number;
  section: string;
  chunk_type: string;
}

export interface SourceRef {
  document_id: string;
  document_title: string;
  page: number | null;
  chunk_id: string | null;
}

export interface CalcInput {
  key: string;
  name: string;
  value: number | null;
  display: string;
  derived: boolean;
  formula: string | null;
  sources: SourceRef[];
}

export interface Calculation {
  id: string;
  kind: "ratio" | "growth" | "metric" | "cagr" | "flag" | "projection";
  key: string;
  name: string;
  category: string;
  company: string;
  period: string;
  formula: string;
  value: number | null;
  unit: string;
  display: string;
  inputs: CalcInput[];
  missing: string[];
  note: string | null;
  severity: "info" | "low" | "medium" | "high" | null;
  change: number | null;
  change_unit: string;
}

export interface DataTable {
  id: string;
  title: string;
  columns: string[];
  rows: string[][];
  note: string | null;
}

export interface ChartSpec {
  id: string;
  title: string;
  kind: "line" | "bar";
  x: string[];
  series: { name: string; data: (number | null)[] }[];
  unit: string;
}

export interface ValidationReport {
  status: "grounded" | "partially_grounded" | "ungrounded" | "not_applicable";
  grounding_score: number | null;
  cited_ids: string[];
  invalid_citations: string[];
  unsupported_numbers: string[];
  uncited_numeric_sentences: number;
  warnings: string[];
}

export interface QueryFilters {
  company_ids: number[];
  document_ids: string[];
  fiscal_years: number[];
  document_types: string[];
}

export type AnswerMode = "generative" | "extractive" | "blocked" | "insufficient_evidence" | "general_knowledge";

export interface ChatResponse {
  conversation_id: string;
  message_id: string;
  answer: string;
  intent: string;
  intent_confidence: number;
  mode: AnswerMode;
  citations: Citation[];
  sources: Citation[];
  calculations: Calculation[];
  tables: DataTable[];
  charts: ChartSpec[];
  validation: ValidationReport;
  followups: string[];
  notes: string[];
  disclaimer: string;
  trace: {
    trace_id?: string;
    total_ms?: number;
    timings_ms?: Record<string, number>;
    usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
    plan?: string[];
    retrieved_chunks?: number;
    llm_model?: string | null;
  };
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface StoredMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  payload: Partial<ChatResponse>;
  created_at: string;
}

export interface ConversationDetail extends ConversationSummary {
  messages: StoredMessage[];
}

export interface PeriodSnapshot {
  period: string;
  fiscal_year: number;
  metrics: Record<string, number | null>;
  ratios: Record<string, number | null>;
}

export interface AnalyzeResponse {
  company: Company;
  analysis_type: string;
  periods: PeriodSnapshot[];
  kpis: Calculation[];
  calculations: Calculation[];
  tables: DataTable[];
  charts: ChartSpec[];
  risk_flags: Calculation[];
  narrative: string | null;
  citations: Citation[];
  notes: string[];
  disclaimer: string;
}

export interface CompareResponse {
  mode: "company" | "period";
  table: DataTable;
  charts: ChartSpec[];
  narrative: string | null;
  citations: Citation[];
  notes: string[];
  disclaimer: string;
}

export interface ReportSummary {
  id: string;
  company_id: number | null;
  company_name: string | null;
  title: string;
  created_at: string;
  meta: { periods?: string[]; source_count?: number; mode?: string; llm_model?: string | null };
}

export interface ReportDetail extends ReportSummary {
  content: string;
  sources: Citation[];
}

export interface DashboardData {
  companies: number;
  documents: number;
  documents_ready: number;
  documents_processing: number;
  documents_failed: number;
  chunks: number;
  facts: number;
  conversations: number;
  reports: number;
  recent_questions: { conversation_id: string; question: string; asked_at: string }[];
  recent_research: ConversationSummary[];
  recent_documents: DocumentItem[];
  spotlight: AnalyzeResponse | null;
  performance: {
    queries?: number;
    errors?: number;
    avg_total_ms?: number | null;
    avg_retrieval_ms?: number | null;
    avg_llm_ms?: number | null;
    avg_grounding_score?: number | null;
    total_tokens?: number;
    intents?: Record<string, number>;
  };
}

export interface SystemInfo {
  app_name: string;
  version: string;
  environment: string;
  llm: { provider: string; model: string | null; configured: boolean; temperature: number; mode: string };
  embeddings: { provider: string; model: string };
  reranker: { provider: string; model: string };
  retrieval: Record<string, number | string>;
  vector_store: { backend: string };
  database: string;
  cache: string;
  auth_enabled: boolean;
  limits: {
    max_upload_mb: number;
    allowed_extensions: string[];
    rate_limit_per_minute: number | null;
    max_query_chars: number;
    disclaimer: string;
  };
}

export interface HealthData {
  status: "ok" | "degraded";
  version: string;
  components: Record<string, { status: string; [key: string]: unknown }>;
}

export interface TraceRow {
  id: string;
  created_at: string;
  intent: string | null;
  status: string;
  total_ms: number;
  timings_ms?: Record<string, number>;
  usage?: { total_tokens: number };
  retrieved_chunks?: number;
  validation_status?: string;
  mode?: string;
}

// Server-sent events emitted by POST /api/chat
export type ChatEvent =
  | { event: "conversation"; data: { conversation_id: string } }
  | { event: "status"; data: { stage: string; message: string } }
  | { event: "meta"; data: { intent: string; confidence: number; plan: string[]; rewritten_query: string | null } }
  | { event: "artifacts"; data: { calculations: Calculation[]; tables: DataTable[]; charts: ChartSpec[] } }
  | { event: "sources"; data: { sources: Citation[] } }
  | { event: "token"; data: { text: string } }
  | { event: "reset"; data: Record<string, never> }
  | { event: "final"; data: ChatResponse }
  | { event: "error"; data: ApiErrorBody };
