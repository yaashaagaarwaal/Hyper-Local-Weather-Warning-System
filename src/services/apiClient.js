// Thin mock-latency wrapper. When the FastAPI backend is ready, replace the
// body of `mockRequest` with a real `fetch(...)` call — callers already
// await a Promise, so no call sites need to change.
const MOCK_LATENCY_MS = 350;

export function mockRequest(factory, { latency = MOCK_LATENCY_MS } = {}) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      try {
        resolve(factory());
      } catch (err) {
        reject(err);
      }
    }, latency);
  });
}

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api";
