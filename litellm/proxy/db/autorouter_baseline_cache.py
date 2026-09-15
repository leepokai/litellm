"""Durable, primary-only revision CAS for opaque auto-router comparison state."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.proxy.db.create_views import SupportsRawQueries
from litellm.proxy.db.routing_prisma_wrapper import writer_wrapper

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_TIMEOUT_SECONDS: Final = 0.25
_READ: Final = 'SELECT revision, state FROM "LiteLLM_AutoRouterBaselineState" WHERE scope = $1'
_INSERT: Final = """
INSERT INTO "LiteLLM_AutoRouterBaselineState" (scope, revision, state)
VALUES ($1, 1, $2) ON CONFLICT (scope) DO NOTHING
"""
_UPDATE: Final = """
UPDATE "LiteLLM_AutoRouterBaselineState" SET revision = revision + 1, state = $2
WHERE scope = $1 AND revision = $3
"""


class _StateRow(TypedDict):
    revision: ReadOnly[int]
    state: ReadOnly[str]


_ROWS: Final = TypeAdapter(tuple[_StateRow, ...])


def _primary(client: PrismaClient) -> SupportsRawQueries:
    return writer_wrapper(client.db)  # pyright: ignore[reportReturnType]  # PrismaWrapper delegates generated methods dynamically.


@dataclass(frozen=True, slots=True)
class BaselineStateSnapshot:
    revision: int
    state: str | None


@dataclass(frozen=True, slots=True)
class BaselineStateFailure:
    reason: str = "state_unavailable"


class PostgresBaselineStateStore:
    def __init__(self, prisma_client: PrismaClient | None) -> None:
        self.prisma_client: Final = prisma_client

    async def read(self, scope: str, now: float) -> BaselineStateSnapshot | BaselineStateFailure:
        if self.prisma_client is None:
            return BaselineStateFailure()
        try:
            primary: Final = _primary(self.prisma_client)
            result: Final = await asyncio.wait_for(primary.query_raw(_READ, scope), timeout=_TIMEOUT_SECONDS)
            rows: Final = _ROWS.validate_python(
                tuple(result),
                strict=True,
            )
            if not rows:
                return BaselineStateSnapshot(0, None)
            if len(rows) != 1 or rows[0]["revision"] < 1:
                return BaselineStateFailure()
            return BaselineStateSnapshot(rows[0]["revision"], rows[0]["state"])
        except Exception:  # noqa: BLE001  # Optional estimates must not replace inference or its original error.
            return BaselineStateFailure()

    async def exchange(
        self, scope: str, before: BaselineStateSnapshot, serialized: str, now: float
    ) -> bool | BaselineStateFailure:
        if self.prisma_client is None:
            return BaselineStateFailure()
        if before.revision < 0 or (before.revision == 0) != (before.state is None):
            return BaselineStateFailure()
        try:
            primary: Final = _primary(self.prisma_client)
            operation: Final = (
                primary.execute_raw(_INSERT, scope, serialized)
                if before.state is None
                else primary.execute_raw(_UPDATE, scope, serialized, before.revision)
            )
            changed: Final = await asyncio.wait_for(operation, timeout=_TIMEOUT_SECONDS)
            return changed == 1 if type(changed) is int and changed in (0, 1) else BaselineStateFailure()
        except Exception:  # noqa: BLE001  # A write with an unknown acknowledgement cannot claim ownership.
            return BaselineStateFailure()
