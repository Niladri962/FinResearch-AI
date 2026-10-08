import { AlertTriangle, Inbox, Loader2 } from "lucide-react";
import type { ReactNode } from "react";

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-ink-soft">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Card({ title, subtitle, actions, children, className = "" }: {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 px-5 pt-4">
          <div>
            {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-muted">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-ink-soft" role="status">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
      {label || "Loading…"}
    </div>
  );
}

export function ErrorBanner({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-line bg-raised p-3 text-sm" role="alert">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-critical" aria-hidden />
      <p className="flex-1 text-ink">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="text-sm font-medium text-accent hover:underline">
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, body, action }: { title: string; body: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center rounded-xl border border-dashed border-line px-6 py-12 text-center">
      <Inbox className="h-7 w-7 text-ink-muted" aria-hidden />
      <h3 className="mt-3 text-sm font-semibold text-ink">{title}</h3>
      <p className="mt-1 max-w-md text-sm text-ink-soft">{body}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

type Tone = "neutral" | "accent" | "good" | "warn" | "serious" | "critical";

const DOT: Record<Tone, string> = {
  neutral: "bg-ink-muted",
  accent: "bg-accent",
  good: "bg-good",
  warn: "bg-warn",
  serious: "bg-serious",
  critical: "bg-critical",
};

/** Status is carried by a coloured dot plus the label, never by text colour alone. */
export function Badge({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border border-line bg-raised px-2 py-0.5 text-xs font-medium text-ink-soft">
      <span className={`h-1.5 w-1.5 rounded-full ${DOT[tone]}`} aria-hidden />
      {children}
    </span>
  );
}

export function Disclaimer({ text }: { text?: string }) {
  return (
    <p className="text-xs leading-relaxed text-ink-muted">
      {text ||
        "FinResearch AI is an analytical research assistant, not a financial advisor. Outputs are generated from the uploaded documents and automated calculations, may contain errors, and should be independently verified. Nothing here is investment advice or a guarantee of future performance."}
    </p>
  );
}

export function CompanySelect({ companies, value, onChange, allowAll = false, label = "Company" }: {
  companies: { id: number; name: string }[];
  value: number | null;
  onChange: (id: number | null) => void;
  allowAll?: boolean;
  label?: string;
}) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <select
        className="input min-w-[14rem]"
        value={value ?? ""}
        onChange={(event) => onChange(event.target.value ? Number(event.target.value) : null)}
      >
        {allowAll ? <option value="">All companies</option> : value === null && <option value="">Select a company</option>}
        {companies.map((company) => (
          <option key={company.id} value={company.id}>
            {company.name}
          </option>
        ))}
      </select>
    </label>
  );
}
