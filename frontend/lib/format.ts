export function compactNumber(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${(value / 1e9).toFixed(1).replace(/\.0$/, "")}B`;
  if (abs >= 1e6) return `${(value / 1e6).toFixed(1).replace(/\.0$/, "")}M`;
  if (abs >= 1e5) return `${(value / 1e3).toFixed(1).replace(/\.0$/, "")}K`;
  if (abs >= 100) return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
  return value.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

export function fullNumber(value: number | null | undefined, unit = ""): string {
  if (value === null || value === undefined) return "Not available";
  const body = value.toLocaleString("en-US", { maximumFractionDigits: 2 });
  if (unit === "%") return `${body}%`;
  if (unit === "x") return `${body}x`;
  return unit ? `${body} ${unit}` : body;
}

export function signed(value: number, unit: string): string {
  const body = Math.abs(value).toLocaleString("en-US", { maximumFractionDigits: 2 });
  const suffix = unit === "pp" ? " pp" : unit;
  return `${value > 0 ? "+" : value < 0 ? "−" : ""}${body}${suffix}`;
}

export function fileSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

export function relativeTime(iso: string): string {
  // Backend timestamps are UTC; SQLite drops the offset, so add it back when missing.
  const date = new Date(/[zZ]|[+-]\d\d:\d\d$/.test(iso) ? iso : `${iso}Z`);
  const seconds = Math.round((Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  if (seconds < 86400 * 7) return `${Math.floor(seconds / 86400)} d ago`;
  return date.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

export function titleCase(value: string): string {
  return value
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function sourceLabel(citation: { document_title: string; page: number; page_end: number | null; section: string }): string {
  const pages = citation.page_end ? `Pages ${citation.page}–${citation.page_end}` : `Page ${citation.page}`;
  return `${citation.document_title}, ${pages}`;
}
