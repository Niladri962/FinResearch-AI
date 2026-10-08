"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "@/lib/api";

export interface AsyncState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/** Fetch on mount (and whenever `deps` change). `pollMs` re-fetches quietly while it is set. */
export function useApi<T>(fetcher: () => Promise<T>, deps: unknown[] = [], pollMs?: number | null): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const load = useCallback(async (quiet: boolean) => {
    if (!quiet) setLoading(true);
    try {
      setData(await fetcherRef.current());
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    if (!pollMs) return;
    const timer = window.setInterval(() => void load(true), pollMs);
    return () => window.clearInterval(timer);
  }, [pollMs, load]);

  return { data, error, loading, reload: () => void load(true) };
}
