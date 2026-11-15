"""In-memory TTL cache with request de-duplication.

Mirrors the pattern already used on the frontend
(src/services/weatherService.js's `requestCache` Map): concurrent callers for
the same key share one in-flight fetch, and a resolved value is reused for
`ttl_seconds` instead of re-fetching.

Deliberately in-process (a plain dict + asyncio locks), not Redis — this is a
single-worker Phase 2 backend and adding a cache service would be an
unnecessary dependency for what this needs. Documented limitation: this cache
is NOT shared across multiple worker processes (e.g. `uvicorn --workers 4`
or multiple gunicorn workers) — each process gets its own cache and will
fetch independently. Fine for Phase 2 and for a single-process deployment;
revisit (Redis or similar) only if/when the API actually runs with multiple
worker processes.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: float):
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, T]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    def peek(self, key: str) -> T | None:
        """Returns the cached value if still fresh, without fetching."""
        entry = self._store.get(key)
        if entry and time.monotonic() - entry[0] < self._ttl:
            return entry[1]
        return None

    async def get_or_fetch(self, key: str, fetch: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        """Returns (value, served_from_cache)."""
        cached = self.peek(key)
        if cached is not None:
            return cached, True

        async with self._locks_guard:
            lock = self._locks.setdefault(key, asyncio.Lock())

        async with lock:
            # Re-check: another coroutine may have populated it while we
            # were waiting for the lock.
            cached = self.peek(key)
            if cached is not None:
                return cached, True

            value = await fetch()
            self._store[key] = (time.monotonic(), value)
            return value, False

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)
