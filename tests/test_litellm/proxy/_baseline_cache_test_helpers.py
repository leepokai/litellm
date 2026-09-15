"""Injected deterministic CAS storage for estimator and lifecycle tests."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Final, Literal

from litellm.proxy.db.autorouter_baseline_cache import BaselineStateFailure, BaselineStateSnapshot


class InMemoryBaselineStore:
    def __init__(self) -> None:
        self.rows: Mapping[str, BaselineStateSnapshot] = MappingProxyType({})
        self.fault: Literal["read", "before", "after", "conflict", "exception"] | None = None
        self.remaining = 0
        self.after_write: Callable[[], None] | None = None
        self.before_write: Callable[[], Awaitable[None]] | None = None
        self.lock = asyncio.Lock()

    def arm(self, fault: Literal["read", "before", "after", "conflict", "exception"]) -> None:
        self.fault = fault
        self.remaining = 4 if fault == "conflict" else 1

    async def read(self, scope: str, now: float) -> BaselineStateSnapshot | BaselineStateFailure:
        if self.fault == "read" and self.remaining:
            self.remaining -= 1
            return BaselineStateFailure()
        return self.rows.get(scope, BaselineStateSnapshot(0, None))

    async def exchange(
        self, scope: str, before: BaselineStateSnapshot, serialized: str, now: float
    ) -> bool | BaselineStateFailure:
        if self.before_write is not None:
            await self.before_write()
        if self.remaining and self.fault in ("before", "conflict", "exception"):
            self.remaining -= 1
            if self.fault == "exception":
                raise ValueError("injected storage exception")
            return BaselineStateFailure() if self.fault == "before" else False
        async with self.lock:
            current: Final = self.rows.get(scope, BaselineStateSnapshot(0, None))
            if current.revision != before.revision:
                return False
            self.rows = MappingProxyType({**self.rows, scope: BaselineStateSnapshot(before.revision + 1, serialized)})
        callback: Final = self.after_write
        if callback is not None:
            self.after_write = None
            callback()
        if self.remaining and self.fault == "after":
            self.remaining -= 1
            return BaselineStateFailure()
        return True
