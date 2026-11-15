import { useCallback, useEffect, useRef, useState } from "react";

// Runs an async fetcher whenever `deps` change, exposing loading/error state.
// `fetcher` should be a stable function reference (e.g. from a service module).
// Pass `{ refreshMs }` to re-run the fetcher on an interval (e.g. live weather
// polling); `refetch()` re-runs it on demand (e.g. a "Retry" button).
export function useAsync(fetcher, deps = [], { refreshMs } = {}) {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [reloadToken, setReloadToken] = useState(0);
  const requestId = useRef(0);

  const refetch = useCallback(() => setReloadToken((t) => t + 1), []);

  useEffect(() => {
    let cancelled = false;
    const currentRequest = ++requestId.current;
    setIsLoading(true);
    setError(null);

    fetcher()
      .then((result) => {
        if (!cancelled && currentRequest === requestId.current) {
          setData(result);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled && currentRequest === requestId.current) {
          setError(err);
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, reloadToken]);

  useEffect(() => {
    if (!refreshMs) return;
    const intervalId = setInterval(refetch, refreshMs);
    return () => clearInterval(intervalId);
  }, [refreshMs, refetch]);

  return { data, isLoading, error, refetch };
}
