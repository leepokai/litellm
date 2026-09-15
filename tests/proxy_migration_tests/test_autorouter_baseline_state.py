"""Real primary/replica routing and CAS against isolated PostgreSQL schemas."""

import asyncio
import os
import time
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Final
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.autorouter_baseline_cache import (
    BaselineStateFailure,
    BaselineStateSnapshot,
    PostgresBaselineStateStore,
)
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from litellm.proxy.utils import PrismaClient, ProxyLogging

_MIGRATION: Final = Path(__file__).parents[2] / (
    "litellm-proxy-extras/litellm_proxy_extras/migrations/20260915010000_add_autorouter_baseline_state/migration.sql"
)


@pytest.fixture
def database() -> Iterator[tuple[str, psycopg.Connection[tuple[object, ...]]]]:
    base: Final = os.environ["DATABASE_URL"].split("?")[0]
    schema: Final = f"baseline_{uuid4().hex}"
    with psycopg.connect(base, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        try:
            connection.execute(_MIGRATION.read_bytes())
            connection.execute(_MIGRATION.read_bytes())
            yield f"{base}?schema={schema}", connection
        finally:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@asynccontextmanager
async def _client(env: pytest.MonkeyPatch, url: str, replica: str | None = None) -> AsyncGenerator[PrismaClient]:
    with env.context() as context:
        context.setenv("DATABASE_URL", url)
        context.delenv("DATABASE_URL_READ_REPLICA", raising=False)
        if replica is not None:
            context.setenv("DATABASE_URL_READ_REPLICA", replica)
        client: Final = PrismaClient(url, ProxyLogging(UserApiKeyCache()))
        try:
            await client.db.connect(timeout=timedelta(seconds=1))
            yield client
        finally:
            await client.db.disconnect()


@pytest.mark.asyncio
async def test_primary_cas_persists_across_clients_and_elapsed_time(
    database: tuple[str, psycopg.Connection[tuple[object, ...]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, connection = database
    async with _client(monkeypatch, url, f"{url}_empty") as first, _client(monkeypatch, url) as second:
        stores: Final = (PostgresBaselineStateStore(first), PostgresBaselineStateStore(second))
        absent: Final = BaselineStateSnapshot(0, None)
        assert await stores[0].read("scope'quoted", 0) == absent
        outcomes: Final = await asyncio.gather(
            *(store.exchange("scope'quoted", absent, str(index), 0) for index, store in enumerate(stores))
        )
        assert outcomes.count(True) == outcomes.count(False) == 1
        before: Final = await stores[1].read("scope'quoted", 1e12)
        assert isinstance(before, BaselineStateSnapshot) and before.revision == 1
        assert before.state == str(outcomes.index(True))
        assert await stores[0].exchange("scope'quoted", before, "next", 1e12) is True
        assert await stores[1].exchange("scope'quoted", before, "stale", 1e12) is False
        assert await stores[0].read("scope'quoted", 1e12) == BaselineStateSnapshot(2, "next")
        assert await stores[1].read("other-scope", 1e12) == absent
        connection.execute('DELETE FROM "LiteLLM_AutoRouterBaselineState"')
        assert await stores[0].exchange("scope'quoted", before, "resurrected", 1e12) is False


@pytest.mark.asyncio
async def test_database_faults_never_become_absence_or_replica_ownership(
    database: tuple[str, psycopg.Connection[tuple[object, ...]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, connection = database
    absent: Final = BaselineStateSnapshot(0, None)
    async with _client(monkeypatch, url) as client:
        store: Final = PostgresBaselineStateStore(client)
        assert await store.exchange("scope", absent, "durable", 0) is True
    async with _client(monkeypatch, "postgresql://unused:unused@127.0.0.1:1/unreachable", url) as degraded:
        assert isinstance(degraded.db, RoutingPrismaWrapper) and degraded.db.writer_unavailable
        unavailable: Final = PostgresBaselineStateStore(degraded)
        assert isinstance(await unavailable.read("scope", 0), BaselineStateFailure)
        assert isinstance(await unavailable.exchange("scope", absent, "wrong", 0), BaselineStateFailure)
    connection.execute('DROP TABLE "LiteLLM_AutoRouterBaselineState"')
    async with _client(monkeypatch, url) as missing:
        for store_without_table in (PostgresBaselineStateStore(missing), PostgresBaselineStateStore(None)):
            assert isinstance(await store_without_table.read("scope", 0), BaselineStateFailure)
            assert isinstance(await store_without_table.exchange("scope", absent, "wrong", 0), BaselineStateFailure)


@pytest.mark.asyncio
async def test_locked_database_is_bounded_and_task_cancellation_propagates(
    database: tuple[str, psycopg.Connection[tuple[object, ...]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, connection = database
    async with _client(monkeypatch, url) as client:
        store: Final = PostgresBaselineStateStore(client)
        with connection.transaction():
            connection.execute('LOCK TABLE "LiteLLM_AutoRouterBaselineState" IN ACCESS EXCLUSIVE MODE')
            started: Final = time.monotonic()
            assert isinstance(await store.read("scope", 0), BaselineStateFailure)
            assert isinstance(
                await store.exchange("scope", BaselineStateSnapshot(0, None), "pending", 0), BaselineStateFailure
            )
            assert time.monotonic() - started < 1.5
            pending: Final = asyncio.create_task(store.read("scope", 0))
            await asyncio.sleep(0.01)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert isinstance(await store.read("scope", 0), BaselineStateSnapshot)
