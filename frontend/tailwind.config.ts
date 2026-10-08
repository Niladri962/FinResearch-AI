import type { Config } from "tailwindcss";

// Colours are CSS variables (see app/globals.css) so light and dark themes swap in one place.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        page: "var(--page)",
        surface: "var(--surface)",
        raised: "var(--raised)",
        line: "var(--line)",
        ink: { DEFAULT: "var(--ink)", soft: "var(--ink-soft)", muted: "var(--ink-muted)" },
        accent: { DEFAULT: "var(--accent)", soft: "var(--accent-soft)", ink: "var(--accent-ink)" },
        good: "var(--good)",
        warn: "var(--warn)",
        serious: "var(--serious)",
        critical: "var(--critical)",
      },
      fontFamily: {
        sans: ["system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      boxShadow: { card: "0 1px 2px rgba(11,11,11,0.04)" },
    },
  },
  plugins: [],
};

export default config;
