// Tiny shared request cache for the home page. Several sections ask for the
// same data (the risk grid, active alerts, one location's weather); without
// this the page would repeat each backend call, and every /risk call stores a
// history row. Entries live for a minute (so the weather desk's 5-minute
// refresh still fetches fresh data) and a failed request is never cached.
const store = new Map();

export function memo(key, loader, ttlMs = 60000) {
  const hit = store.get(key);
  if (hit && Date.now() - hit.at < ttlMs) return hit.promise;
  const promise = Promise.resolve()
    .then(loader)
    .catch((err) => {
      store.delete(key);
      throw err;
    });
  store.set(key, { at: Date.now(), promise });
  return promise;
}
