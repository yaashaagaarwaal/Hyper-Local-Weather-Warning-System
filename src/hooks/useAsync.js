import { useEffect, useRef, useState } from "react";

// Runs an async fetcher whenever `deps` change, exposing loading/error state.
// `fetcher` should be a stable function reference (e.g. from a service module).
export function useAsync(fetcher, deps = []) {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const requestId = useRef(0);

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
  }, deps);

  return { data, isLoading, error };
}
