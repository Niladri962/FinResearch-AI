"use client";

import {
  BarChart3,
  Building2,
  FileText,
  GitCompareArrows,
  LayoutDashboard,
  LineChart,
  Menu,
  MessagesSquare,
  ScrollText,
  Settings,
  X,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import { apiConfigProblem } from "@/lib/api";

const NAV = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/companies", label: "Companies", icon: Building2 },
  { href: "/documents", label: "Documents", icon: FileText },
  { href: "/chat", label: "Research Chat", icon: MessagesSquare },
  { href: "/analysis", label: "Financial Analysis", icon: LineChart },
  { href: "/compare", label: "Comparisons", icon: GitCompareArrows },
  { href: "/reports", label: "Reports", icon: ScrollText },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => setProblem(apiConfigProblem()), []);
  const fullBleed = pathname.startsWith("/chat");

  const nav = (
    <nav className="flex flex-col gap-0.5 px-3" aria-label="Main">
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            onClick={() => setOpen(false)}
            aria-current={active ? "page" : undefined}
            className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors ${
              active ? "bg-accent-soft font-medium text-accent-ink" : "text-ink-soft hover:bg-page hover:text-ink"
            }`}
          >
            <Icon className="h-4 w-4" aria-hidden />
            {label}
          </Link>
        );
      })}
    </nav>
  );

  const brand = (
    <div className="flex items-center gap-2.5 px-5 py-5">
      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-white">
        <BarChart3 className="h-4 w-4" aria-hidden />
      </span>
      <div className="leading-tight">
        <p className="text-sm font-semibold text-ink">FinResearch AI</p>
        <p className="text-[11px] text-ink-muted">Financial research &amp; analysis</p>
      </div>
    </div>
  );

  return (
    <div className="flex h-screen overflow-hidden">
      <aside className="no-print hidden w-60 shrink-0 flex-col border-r border-line bg-surface lg:flex">
        {brand}
        {nav}
        <p className="mt-auto px-5 py-4 text-[11px] leading-relaxed text-ink-muted">
          Research assistant, not a financial advisor. Verify outputs independently.
        </p>
      </aside>

      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-64 border-r border-line bg-surface">
            <div className="flex items-center justify-between pr-3">
              {brand}
              <button onClick={() => setOpen(false)} aria-label="Close menu" className="rounded-lg p-2 text-ink-soft hover:bg-page">
                <X className="h-4 w-4" />
              </button>
            </div>
            {nav}
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="no-print flex items-center gap-3 border-b border-line bg-surface px-4 py-3 lg:hidden">
          <button onClick={() => setOpen(true)} aria-label="Open menu" className="rounded-lg p-1.5 text-ink-soft hover:bg-page">
            <Menu className="h-5 w-5" />
          </button>
          <span className="text-sm font-semibold">FinResearch AI</span>
        </header>
        {problem && (
          <p className="no-print border-b border-line bg-raised px-5 py-2.5 text-sm text-ink" role="alert">
            <span className="mr-2 inline-block h-2 w-2 rounded-full bg-critical align-middle" aria-hidden />
            {problem}
          </p>
        )}
        <main className={`min-h-0 flex-1 ${fullBleed ? "overflow-hidden" : "overflow-y-auto"}`}>
          {fullBleed ? children : <div className="mx-auto max-w-6xl px-5 py-7 sm:px-8">{children}</div>}
        </main>
      </div>
    </div>
  );
}
