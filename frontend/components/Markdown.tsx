"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const MARKER = /\[((?:[SCT]\d+)(?:\s*,\s*(?:[SCT]\d+))*)\]/g;

/** Turn `[S1, C2]` markers into links the renderer below shows as citation chips. */
function linkCitations(text: string): string {
  return text.replace(MARKER, (_match, ids: string) =>
    ids
      .split(",")
      .map((id) => id.trim())
      .map((id) => `[${id}](#cite-${id})`)
      .join(" "),
  );
}

export function Markdown({ content, onCite }: { content: string; onCite?: (id: string) => void }) {
  return (
    <div className="prose-fin">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children }) {
            if (href?.startsWith("#cite-")) {
              const id = href.slice(6);
              const kind = id[0] === "S" ? "Source" : id[0] === "C" ? "Computed metric" : "Computed table";
              return (
                <button
                  type="button"
                  onClick={() => onCite?.(id)}
                  title={`${kind} ${id}`}
                  className="mx-0.5 inline-flex h-[18px] items-center rounded border border-line bg-accent-soft px-1 align-baseline font-mono text-[10.5px] font-medium text-accent-ink hover:border-accent"
                >
                  {children}
                </button>
              );
            }
            return (
              <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent underline">
                {children}
              </a>
            );
          },
          table({ children }) {
            return (
              <div className="table-wrap">
                <table>{children}</table>
              </div>
            );
          },
        }}
      >
        {linkCitations(content)}
      </ReactMarkdown>
    </div>
  );
}
