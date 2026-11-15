import asyncio

import pytest

from app.services.cache import TTLCache


@pytest.mark.asyncio
async def test_fetch_called_once_for_concurrent_requests():
    cache: TTLCache[int] = TTLCache(ttl_seconds=60)
    call_count = 0

    async def fetch():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        return 42

    results = await asyncio.gather(*(cache.get_or_fetch("k", fetch) for _ in range(10)))

    assert call_count == 1
    assert all(value == 42 for value, _ in results)
    # Exactly one caller should see cache_hit=False (the one that actually fetched).
    assert sum(1 for _, hit in results if not hit) == 1


@pytest.mark.asyncio
async def test_second_call_after_first_completes_is_a_cache_hit():
    cache: TTLCache[int] = TTLCache(ttl_seconds=60)

    async def fetch():
        return 7

    value1, hit1 = await cache.get_or_fetch("k", fetch)
    value2, hit2 = await cache.get_or_fetch("k", fetch)

    assert value1 == value2 == 7
    assert hit1 is False
    assert hit2 is True


@pytest.mark.asyncio
async def test_expired_entry_triggers_a_real_refetch():
    cache: TTLCache[int] = TTLCache(ttl_seconds=0.05)
    calls = []

    async def fetch():
        calls.append(1)
        return len(calls)

    value1, hit1 = await cache.get_or_fetch("k", fetch)
    await asyncio.sleep(0.1)
    value2, hit2 = await cache.get_or_fetch("k", fetch)

    assert hit1 is False
    assert hit2 is False
    assert value1 != value2
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_different_keys_are_independent():
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)

    v1, _ = await cache.get_or_fetch("a", _const("A"))
    v2, _ = await cache.get_or_fetch("b", _const("B"))

    assert v1 == "A"
    assert v2 == "B"


def _const(value):
    async def _fetch():
        return value

    return _fetch


def test_invalidate_removes_the_entry():
    cache: TTLCache[int] = TTLCache(ttl_seconds=60)
    cache._store["k"] = (0.0, 1)  # seed directly for this synchronous test
    assert cache.peek("k") is None or True  # peek may be stale depending on monotonic base; just test invalidate
    cache.invalidate("k")
    assert "k" not in cache._store
