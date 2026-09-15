import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.llms.anthropic.prompt_cache_prediction import NativePredictionTarget
from litellm.proxy.db.autorouter_baseline_cache import BaselineStateSnapshot
from litellm.proxy.spend_tracking.autorouter_baseline_cache import (
    BaselineCacheEstimate,
    BaselineCacheEstimator,
    BaselineReservation,
    _State,  # pyright: ignore[reportPrivateUsage]  # preserve the typed atomic state transition in the fault store
)
from litellm.types.utils import Usage
from tests.test_litellm.proxy._baseline_cache_test_helpers import InMemoryBaselineStore

_MODEL: Final = "claude-sonnet-5"
_TARGET: Final = NativePredictionTarget(_MODEL, "test-provider-key", "https://configured-native-provider.test")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
pytestmark: Final = [pytest.mark.usefixtures("local_model_cost_map"), pytest.mark.asyncio]


@dataclass
class _Clock:
    now: float = 10000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class _Counter:
    unavailable: bool = False
    calls: int = 0

    async def __call__(self, model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        self.calls += 1
        return None if self.unavailable else json.dumps(_JSON_OBJECT.validate_python(body)).count("token ")


def _block(tokens: int = 0, ttl: str | None = None, *, text: str = "") -> dict[str, JsonValue]:
    block: Final[dict[str, JsonValue]] = {"type": "text", "text": text or "token " * tokens}
    if ttl is not None:
        block["cache_control"] = {"type": "ephemeral", "ttl": ttl}
    return block


def _native(*blocks: dict[str, JsonValue], target: NativePredictionTarget = _TARGET) -> httpx.Request:
    return httpx.Request(
        "POST", f"{target.api_base}/v1/messages",
        headers=MappingProxyType({"anthropic-version": "2023-06-01", "x-api-key": target.api_key}),
        json={"model": _MODEL, "max_tokens": 10, "messages": [{"role": "user", "content": list(blocks)}]},
    )


def _wire(
    ttl: str = "1h", *, growth: int = 0, changed: bool = False, target: NativePredictionTarget = _TARGET
) -> httpx.Request:
    blocks: Final = [_block(text=("changed " if changed else "") + "token " * 6000)]
    if growth:
        blocks.append(_block(growth))
    return _native(*blocks, _block(ttl=ttl, text="end"), target=target)


@dataclass
class _Rig:
    clock: _Clock = field(default_factory=_Clock)
    counter: _Counter = field(default_factory=_Counter)
    faults: InMemoryBaselineStore = field(default_factory=InMemoryBaselineStore)
    estimator: BaselineCacheEstimator = field(init=False)

    def __post_init__(self) -> None:
        self.estimator = BaselineCacheEstimator(self.faults, self.clock, self.counter)

    def prepare(
        self, request: str, *, caller: str = "caller", session: str = "session",
        target: NativePredictionTarget = _TARGET,
    ) -> BaselineReservation:
        reservation: Final = self.estimator.prepare(
            caller_key_hash=caller, session_id=session, router_id="router",
            baseline_deployment_id="baseline", target=target, request_id=request,
        )
        assert isinstance(reservation, BaselineReservation)
        return reservation

    async def reserve(self, request: str) -> BaselineReservation:
        reservation: Final = self.prepare(request)
        assert await self.estimator.reserve(reservation) is None
        return reservation

    async def finish(
        self, reservation: BaselineReservation, wire: httpx.Request | None = None,
        *, available_at: float | None = None, observed_usage: Usage | None = None,
        completed: bool = True, cache_hit: bool = False, observed_cache_tokens: int = 0,
    ) -> BaselineCacheEstimate:
        return await self.estimator.finalize(
            reservation, wire=wire if wire is not None else _wire(), request_started_at=reservation.reserved_at,
            available_at=self.clock.now if available_at is None else available_at, observed_usage=observed_usage,
            completed=completed, cache_hit=cache_hit, observed_cache_tokens=observed_cache_tokens,
        )

    async def run(
        self, request: str, wire: httpx.Request | None = None, *, observed_usage: Usage | None = None,
    ) -> BaselineCacheEstimate:
        reservation: Final = await self.reserve(request)
        self.clock.advance(0.1)
        return await self.finish(reservation, wire, observed_usage=observed_usage)

    async def state(self, reservation: BaselineReservation) -> _State:
        snapshot: Final = await self.estimator.store.read(reservation.scope, self.clock.now)
        assert isinstance(snapshot, BaselineStateSnapshot) and snapshot.state is not None
        return _State.model_validate_json(snapshot.state)

    def fork(self) -> "_Rig":
        return _Rig(clock=self.clock, faults=self.faults)


@pytest.fixture
def rig() -> _Rig:
    return _Rig()


@pytest.mark.parametrize(
    "ttl,duration,bucket",
    (("5m", 300, "cache_creation_5m_input_tokens"), ("1h", 3600, "cache_creation_1h_input_tokens")),
)
async def test_established_expiry_never_receives_a_hypothetical_read(
    rig: _Rig, ttl: str, duration: int, bucket: str
) -> None:
    first: Final = await rig.run("first", _wire(ttl))
    assert (first.status, first.reason) == ("unknown", "history_unavailable")
    rig.clock.now = 10001.0
    counted: Final = rig.counter.calls
    warm: Final = await rig.run("warm", _wire(ttl))
    assert rig.counter.calls == counted
    assert warm.cache_read_input_tokens == 6000
    assert warm.cache_creation_1h_input_tokens == warm.cache_creation_5m_input_tokens == 0
    rig.clock.now = 10001.0 + duration
    expired: Final = await rig.run("expired", _wire(ttl))
    assert (expired.status, expired.reason) == ("estimated", "cache_prefix_expired")
    assert expired.cache_read_input_tokens == 0
    assert expired.metadata()[bucket] == 6000
    usage: Final = expired.usage(10)
    assert usage is not None and usage.prompt_tokens == 6000 and usage.total_tokens == 6010


@pytest.mark.parametrize("ttl,duration", (("5m", 300), ("1h", 3600)))
@pytest.mark.parametrize("cancelled", (False, True))
async def test_invalidation_keeps_unknown_cache_effects_through_latest_retry_completion(
    rig: _Rig, ttl: str, duration: int, cancelled: bool
) -> None:
    await rig.run("seed", _wire(ttl))
    rig.clock.now = 10001.0
    assert (await rig.run("warm", _wire(ttl))).cache_read_input_tokens == 6000
    rig.clock.now = 10005.0
    failed: Final = await rig.reserve("retrying")
    if cancelled:
        await rig.estimator.cancel(failed)
    calls: Final = rig.counter.calls
    for at, reason in ((10010.0, "upstream_request_failed"), (10010.0 + duration, "retried_upstream_request")):
        rig.clock.now = at
        invalidated: BaselineCacheEstimate = await rig.estimator.invalidate(failed, reason)
        assert (invalidated.status, invalidated.reason) == ("unknown", reason)
    assert rig.counter.calls == calls
    rig.clock.now = 10020.0 + duration
    following: Final = await rig.run("following", _wire(ttl))
    assert (following.status, following.reason) == ("unknown", "history_unavailable")
    assert following.cache_read_input_tokens is None
    rig.clock.now = 10020.0 + 2 * duration
    expired: Final = await rig.run("expired", _wire(ttl))
    assert (expired.status, expired.reason) == ("estimated", "cache_prefix_expired")
    assert expired.cache_read_input_tokens == 0


async def test_prefix_change_is_cold_after_history_horizon_and_unknown_before_it(rig: _Rig) -> None:
    await rig.run("first")
    assert (await rig.run("changed-early", _wire(changed=True))).status == "unknown"
    rig.clock.now = 13601.0
    changed: Final = await rig.run("changed", _wire(growth=2000))
    assert changed.status == "estimated"
    assert changed.cache_read_input_tokens == 0
    assert changed.cache_creation_1h_input_tokens == 8000


async def test_lookback_reads_prior_marker_and_writes_only_growth(rig: _Rig) -> None:
    await rig.run("first")
    rig.clock.now = 13600.0
    await rig.run("cold")
    growing_wire: Final = _native(_block(6000), _block(text="end"), _block(2000, "1h"))
    grown: Final = await rig.run("grown", growing_wire)
    assert grown.status == "estimated"
    assert grown.cache_read_input_tokens == 6000
    assert grown.cache_creation_1h_input_tokens == 2000
    rig.clock.now = 17200.05
    assert (await rig.run("old-prefix")).cache_read_input_tokens == 6000


async def test_parallel_pending_requests_and_late_completions_do_not_self_hit_or_regress_refresh(rig: _Rig) -> None:
    first: Final = await rig.reserve("first")
    rig.clock.now = 10001.0
    second: Final = await rig.reserve("second")
    rig.clock.now = 10002.0
    second_result: Final = await rig.finish(second)
    assert second_result.reason == "pending_request"
    assert (await rig.finish(first)).cache_read_input_tokens is None
    assert (await rig.finish(second, _wire(changed=True))) == second_result
    rig.clock.now = 13600.5
    assert (await rig.run("still-warm")).cache_read_input_tokens == 6000


@pytest.mark.parametrize(
    "caller,session,target",
    (
        ("other", "session", _TARGET),
        ("caller", "other", _TARGET),
        ("caller", "session", NativePredictionTarget(_MODEL, "other-key", _TARGET.api_base)),
        ("caller", "session", NativePredictionTarget(_MODEL, "test-provider-key", "https://other.test")),
    ),
)
async def test_scope_isolates_callers_sessions_and_baseline_credentials(
    rig: _Rig, caller: str, session: str, target: NativePredictionTarget
) -> None:
    await rig.run("first")
    isolated: Final = rig.prepare("second", caller=caller, session=session, target=target)
    assert await rig.estimator.reserve(isolated) is None
    assert (await rig.finish(isolated, _wire(target=target))).reason == "history_unavailable"


@pytest.mark.parametrize("baseline_base,wire_base,wire_key,supported", (
    ("https://gateway.example", "https://gateway.example", _TARGET.api_key, True),
    ("https://gateway.example/v1/messages", "https://gateway.example", _TARGET.api_key, True),
    ("https://api.anthropic.com", "https://api.anthropic.com", _TARGET.api_key, True),
    ("https://other.example", "https://gateway.example", _TARGET.api_key, False),
    ("https://gateway.example/other", "https://gateway.example", _TARGET.api_key, False),
    ("https://api.anthropic.com", "https://gateway.example", _TARGET.api_key, False),
    ("https://gateway.example", "https://gateway.example", "other-account", False),
    ("https://gateway.example", "https://gateway.example", None, False),
), ids=("gateway-root", "messages-path", "first-party", "other-host", "other-path",
        "gateway-to-first-party", "other-key", "missing-key"))
async def test_baseline_count_uses_only_the_captured_request_recipient(
    rig: _Rig, baseline_base: str, wire_base: str, wire_key: str | None, supported: bool
) -> None:
    target: Final = NativePredictionTarget("claude-opus-5", _TARGET.api_key, baseline_base)
    reservation: Final = rig.prepare("recipient", target=target)
    assert await rig.estimator.reserve(reservation) is None
    wire: Final = httpx.Request(
        "POST", wire_base + "/v1/messages", content=_native(_block(100, "1h")).content,
        headers={"x-api-key": wire_key} if wire_key is not None else {},
    )
    result: Final = await rig.finish(reservation, wire)
    assert result.status == ("estimated" if supported else "unknown")
    assert (rig.counter.calls > 0) is supported
    state: Final = await rig.state(reservation)
    assert not state.pending and not state.versions
    if not supported:
        assert result.reason == "unsupported_baseline_recipient"


@pytest.mark.parametrize("blocks,observed,reason,buckets", (
    ((_block(1, "1h"),), 0, "below_cache_minimum", (1, 0, 0, 0)),
    ((_block(6000),), 0, "no_cache_breakpoints", (6000, 0, 0, 0)),
    ((_block(6000),), 6000, "implicit_cache_without_breakpoints", (None, None, None, None)),
    ((_block(5000, "1h"), _block(2000, "5m"), _block(100)), 0, "cache_prefix_cold", (100, 0, 2000, 5000)),
), ids=("below-minimum", "unmarked-ordinary", "unmarked-provider-cache", "mixed-ttl"))
async def test_cache_bucket_partition_and_implicit_cache_inputs(
    rig: _Rig, blocks: tuple[dict[str, JsonValue], ...], observed: int,
    reason: str, buckets: tuple[int | None, ...],
) -> None:
    if reason == "cache_prefix_cold":
        await rig.run("seed")
        rig.clock.advance(3600)
    reservation: Final = await rig.reserve("partition")
    calls: Final = rig.counter.calls
    result: Final = await rig.finish(reservation, _native(*blocks), observed_cache_tokens=observed)
    assert result.reason == reason and result.status == ("unknown" if observed else "estimated")
    assert (result.input_tokens, result.cache_read_input_tokens,
            result.cache_creation_5m_input_tokens, result.cache_creation_1h_input_tokens) == buckets
    if observed:
        assert rig.counter.calls == calls


async def test_ttl_changes_remain_unknown_until_every_possible_refreshed_entry_expires(rig: _Rig) -> None:
    await rig.run("first", _wire("1h"))
    assert (await rig.run("warm", _wire("1h"))).cache_read_input_tokens == 6000
    for request, at in (("changed", 10002.0), ("ambiguous", 10303.0), ("refreshed", 13602.5)):
        rig.clock.now = at
        changed: BaselineCacheEstimate = await rig.run(request, _wire("5m"))
        assert (changed.status, changed.reason) == ("unknown", "cache_ttl_changed")
    rig.clock.now = 17204.0
    expired: Final = await rig.run("expired", _wire("5m"))
    assert expired.status == "estimated"
    assert expired.cache_read_input_tokens == 0
    assert expired.cache_creation_5m_input_tokens == 6000


async def test_pruned_pending_reservation_cannot_advance_cache_state(rig: _Rig) -> None:
    first: Final = await rig.reserve("first")
    for index in range(256):
        await rig.reserve(f"pending-{index}")
    assert (await rig.finish(first)).reason == "reservation_unavailable"


async def test_long_running_request_keeps_the_cache_history_needed_at_its_start(rig: _Rig) -> None:
    await rig.run("seed", _wire("5m"))
    rig.clock.now = 10001.0
    delayed: Final = await rig.reserve("delayed")
    rig.clock.now = 15000.0
    result: Final = await rig.finish(delayed, _wire("5m"))
    assert result.status == "estimated"
    assert result.cache_read_input_tokens == 6000


async def test_durable_store_survives_estimator_restart_and_corruption_never_looks_absent(rig: _Rig) -> None:
    first: Final = await rig.reserve("first")
    await rig.finish(first)
    second: Final = rig.fork()
    assert (await second.run("second")).cache_read_input_tokens == 6000
    snapshot: Final = await rig.faults.read(first.scope, rig.clock.now)
    assert isinstance(snapshot, BaselineStateSnapshot)
    assert await rig.faults.exchange(first.scope, snapshot, "invalid state", rig.clock.now)
    refused: Final = await second.estimator.reserve(second.prepare("corrupt"))
    assert refused is not None and refused.reason == "state_corrupt"


@pytest.mark.parametrize("operation", ("reserve-terminal", "reserve-cancel", "finalize", "invalidate", "cancel"))
@pytest.mark.parametrize("fault", ("read", "before", "after", "conflict"))
async def test_storage_failure_retires_exact_request_on_recovery(
    rig: _Rig, operation: str, fault: Literal["read", "before", "after", "conflict"]
) -> None:
    await rig.run("seed")
    failed: Final = rig.prepare("failed")
    if not operation.startswith("reserve-"):
        assert await rig.estimator.reserve(failed) is None
    rig.faults.arm(fault)
    if operation.startswith("reserve-"):
        result: Final = await rig.estimator.reserve(failed)
        assert result is not None and result.status == "unknown"
        if operation == "reserve-cancel":
            await rig.estimator.cancel(failed)
        else:
            assert (await rig.run("during-unacknowledged")).reason == "pending_request"
            rig.clock.advance(10)
            await rig.estimator.invalidate(failed, "registration_failed", completed=True)
    elif operation == "finalize":
        assert (await rig.finish(failed)).status == "unknown"
    elif operation == "invalidate":
        assert (await rig.estimator.invalidate(failed, "final_response_unknown")).status == "unknown"
        rig.clock.advance(3601)
        rig.estimator.defer_cancel(rig.prepare("other-scope", session="other"))
    else:
        await rig.estimator.cancel(failed)
    recovered: Final = await rig.run("recovered")
    if operation in ("cancel", "reserve-cancel"):
        assert recovered.cache_read_input_tokens == 6000
    else:
        assert recovered.reason == ("cache_prefix_cold" if operation == "invalidate" else "history_unavailable")
    if operation == "invalidate":
        await rig.estimator.invalidate(failed, "late_active_callback", completed=False)
        assert (await rig.run("after-late-callback")).reason == "history_unavailable"
    state: Final = await rig.state(failed)
    assert not any(item.request_id == failed.request_id for item in state.pending)
    assert any(item.request_id == failed.request_id for item in state.completed)


async def test_failed_old_finalizer_preserves_retry_owned_by_another_process(rig: _Rig) -> None:
    await rig.run("seed")
    retrying: Final = await rig.reserve("retry-race")
    observer: Final = rig.fork()
    rig.faults.arm("after")
    await rig.estimator.invalidate(retrying, "retry_active", completed=False)
    assert (await observer.finish(retrying)).reason == "retry_active"
    assert (await observer.run("during-retry")).reason == "pending_request"
    rig.clock.advance(10)
    await rig.estimator.invalidate(retrying, "retry_terminal", completed=True)
    assert (await observer.run("after-retry-terminal")).reason == "history_unavailable"


@pytest.mark.parametrize("unsupported", (False, True))
async def test_active_invalidation_supersedes_acknowledged_provisional_finish_but_not_logical_terminal(
    rig: _Rig, unsupported: bool,
) -> None:
    completed: Final = await rig.reserve("provisional")
    valid: Final = _wire()
    wire: Final = httpx.Request("POST", valid.url, headers=valid.headers, content="invalid") if unsupported else valid
    await rig.finish(completed, wire)
    await rig.estimator.invalidate(completed, "new_retry", completed=False)
    assert (await rig.run("during-new-retry")).reason == "pending_request"
    await rig.estimator.invalidate(completed, "logical_terminal", completed=True)
    await rig.estimator.invalidate(completed, "late_failure_callback", completed=False)
    state: Final = await rig.state(completed)
    assert not any(item.request_id == completed.request_id for item in state.pending)
    final: Final = next(item for item in state.completed if item.request_id == completed.request_id)
    assert final.terminal and final.estimate.status == "unknown"


async def test_acknowledged_older_repair_cannot_erase_new_terminal_debt(rig: _Rig) -> None:
    active: Final = await rig.reserve("overlap")
    rig.estimator.defer(active, "still_active", completed=False)

    def complete_while_acknowledging() -> None:
        rig.clock.advance(10)
        rig.estimator.defer(active, "now_terminal", completed=True)

    rig.faults.after_write = complete_while_acknowledging
    intermediate: Final = await rig.estimator.reserve(active)
    assert intermediate is not None and intermediate.reason == "still_active"
    assert (await rig.run("after-overlap")).reason == "history_unavailable"
    state: Final = await rig.state(active)
    assert not any(item.request_id == active.request_id for item in state.pending)
    completed: Final = next(item for item in state.completed if item.request_id == active.request_id)
    assert completed.estimate.reason == "now_terminal"
    assert state.uncertain_before >= 10010.0


@pytest.mark.parametrize("early_result", ("pending", "completed"))
async def test_idempotent_results_publish_other_cleanup_without_removing_live_requests(
    rig: _Rig, early_result: str
) -> None:
    existing: Final = await rig.reserve("existing")
    if early_result == "completed":
        await rig.finish(existing)
    abandoned: Final = await rig.reserve("abandoned")
    unrelated: Final = await rig.reserve("unrelated-live")
    rig.estimator.defer(abandoned, "abandoned", completed=True)
    if early_result == "pending":
        assert await rig.estimator.reserve(existing) is None
    else:
        duplicate: Final = await rig.finish(existing)
        assert duplicate.status == "unknown"
    state: Final = await rig.state(existing)
    pending_ids: Final = frozenset(item.request_id for item in state.pending)
    assert abandoned.request_id not in pending_ids
    assert unrelated.request_id in pending_ids
    assert (existing.request_id in pending_ids) == (early_result == "pending")


@pytest.mark.parametrize("loss", (
    "restart", "equal", "delayed", "delayed-equal", "old-comparison", "old-equal", "fresh",
))
async def test_lost_unpublished_repairs_preserve_pending_uncertainty(rig: _Rig, loss: str) -> None:
    pending: Final = await rig.reserve("lost-terminal")
    if loss == "restart":
        rig.faults.arm("before")
        await rig.estimator.invalidate(pending, "terminal")
    else:
        prepared: Final = rig.prepare("new-owner", session="new-scope")
        if loss.startswith("old"):
            assert await rig.estimator.reserve(prepared) is None
        rig.clock.advance(0 if loss.endswith("-equal") else 1)
        rig.estimator.defer(pending, "lost_repair", completed=True)
        for index in range(1024):
            rig.estimator.defer(rig.prepare(f"overflow-{index}", session=f"scope-{index}"), "uncertain", completed=True)
        assert len(rig.estimator.repairs) == 1024
        rig.clock.advance(0 if loss == "equal" else 1)
        new_scope: Final = prepared if loss.startswith("delayed") else rig.prepare("new-owner", session="new-scope")
        assert await rig.estimator.reserve(new_scope) is None
        result: Final = await rig.finish(new_scope, observed_usage=_observed())
        assert (result.provenance == "observed_initial") == (loss == "fresh")
        assert result.comparison_started_at == (prepared.reserved_at if loss.startswith("old") else rig.clock.now)
    observer: Final = rig.fork() if loss == "restart" else rig
    assert (await observer.run("after-repair-loss")).reason == "pending_request"
    rig.clock.advance(7201)
    assert (await observer.run("after-retention")).reason == "history_unavailable"


@pytest.mark.parametrize("cancel_last", (False, True))
async def test_cancel_and_invalidation_debt_order_never_resurrects_pending(rig: _Rig, cancel_last: bool) -> None:
    await rig.run("seed")
    reservation: Final = await rig.reserve("ordered")
    if cancel_last:
        rig.estimator.defer(reservation, "registration_unknown", completed=False)
        rig.estimator.defer_cancel(reservation)
    else:
        rig.estimator.defer_cancel(reservation)
        rig.estimator.defer(reservation, "late_upstream_evidence", completed=False)
    following: Final = await rig.run("after-ordered")
    assert following.reason != "pending_request"
    if cancel_last:
        assert following.cache_read_input_tokens == 6000
    else:
        assert following.reason == "history_unavailable"


@pytest.mark.parametrize("order", ("success-first", "fault-first", "overflow"))
@pytest.mark.parametrize("available_at", (10001.0, 10002.0, 10003.0), ids=("before", "equal", "after"))
async def test_independent_success_cannot_cross_fault_cutoff_by_finalizer_order(
    rig: _Rig, order: str, available_at: float
) -> None:
    invalidator: Final = rig.fork()
    seed: Final = await rig.reserve("delayed-independent-success")
    rig.clock.now = 10001.0
    failed: Final = await invalidator.reserve("terminal-failure")
    rig.clock.now = 10002.0
    invalidator.estimator.defer(failed, "failed_request", completed=True)
    if order == "overflow":
        for index in range(1024):
            invalidator.estimator.defer(rig.prepare(f"overflow-{index}", session=f"scope-{index}"), "uncertain", completed=True)
    if order == "fault-first":
        assert await invalidator.estimator.reserve(failed) is not None
    rig.clock.now = 10004.0
    await rig.finish(seed, available_at=available_at)
    if order == "overflow":
        await invalidator.estimator.cancel(failed)
    elif order == "success-first":
        assert await invalidator.estimator.reserve(failed) is not None
    rig.clock.now = 10005.0
    following: Final = await rig.run("after-terminal-and-independent-success")
    if available_at > 10002.0:
        assert following.cache_read_input_tokens == 6000
    else:
        assert following.reason == "history_unavailable"
    assert (await rig.run("matching-after-restored-evidence")).cache_read_input_tokens == 6000


def _observed(write_5m: int = 0, write_1h: int = 0, read: int = 0) -> Usage:
    return Usage(
        prompt_tokens=6007, completion_tokens=13, total_tokens=6020,
        prompt_tokens_details={
            "text_tokens": 6007 - write_5m - write_1h - read, "cached_tokens": read,
            "cache_creation_tokens": write_5m + write_1h,
            "cache_creation_token_details": {
                "ephemeral_5m_input_tokens": write_5m, "ephemeral_1h_input_tokens": write_1h,
            },
        },
        completion_tokens_details={"reasoning_tokens": 3, "text_tokens": 10},
        cache_read_input_tokens=read, cache_creation_input_tokens=write_5m + write_1h,
        server_tool_use={"web_search_requests": 1}, cost=0.123,
        inference_geo="us", speed="fast",
    )


@pytest.mark.parametrize("write_5m,write_1h,read", ((6000, 0, 0), (0, 6000, 0), (0, 0, 6000), (0, 0, 0)))
@pytest.mark.parametrize("count_mode", ("available", "unavailable", "unsupported_headers"))
async def test_initial_observation_retains_full_usage_independently_of_counting_and_replay(
    rig: _Rig, write_5m: int, write_1h: int, read: int, count_mode: str,
) -> None:
    initial: Final = await rig.reserve("initial")
    rig.counter.unavailable = count_mode == "unavailable"
    wire: Final = _wire()
    if count_mode == "unsupported_headers":
        wire.headers["anthropic-beta"] = "claude-code-20250219"
    observed: Final = _observed(write_5m, write_1h, read)
    result: Final = await rig.finish(initial, wire, observed_usage=observed)
    assert (result.status, result.provenance, result.reason) == (
        "estimated", "observed_initial", "observed_initial_baseline",
    )
    assert result.usage(999) == observed and result.usage(999) is not observed
    assert result.metadata()["comparison_started_at"] == initial.reserved_at
    assert "observed_usage" not in result.metadata()
    replay: Final = await rig.fork().finish(initial)
    assert replay == result and replay.usage(999) == observed
    state: Final = await rig.state(initial)
    assert not state.first_eligible and bool(state.versions) == (count_mode == "available")
    if count_mode == "unsupported_headers":
        assert (await rig.run("modeled-with-header", wire)).reason == "unsupported_request_headers"
    rig.counter.unavailable = False
    assert (await rig.run("unseen-longer-prefix", _wire(growth=1000))).reason == "history_unavailable"


@pytest.mark.parametrize("disqualifier", (
    "overlap", "no-evidence", "different-model", "incomplete-usage", "cancel", "retry", "failed-admission",
    "failed-finalize", "count-unavailable", "incomplete", "gateway-cache",
))
async def test_only_exclusive_healthy_original_owner_can_anchor(rig: _Rig, disqualifier: str) -> None:
    initial: Final = rig.prepare("initial")
    if disqualifier == "failed-admission":
        rig.faults.arm("after")
    admitted: Final = await rig.estimator.reserve(initial)
    assert (admitted is not None) == (disqualifier == "failed-admission")
    if disqualifier == "overlap":
        await rig.fork().reserve("concurrent")
    elif disqualifier == "cancel":
        await rig.estimator.cancel(initial)
    elif disqualifier == "retry":
        await rig.estimator.invalidate(initial, "retry", completed=False)
    valid: Final = _wire()
    wire: Final = httpx.Request(
        "POST", valid.url, headers=valid.headers,
        json={**_JSON_OBJECT.validate_json(valid.content), "model": "claude-opus-4-6"},
    ) if disqualifier == "different-model" else valid
    rig.counter.unavailable = disqualifier == "count-unavailable"
    candidate: Final = None if disqualifier in ("no-evidence", "count-unavailable") else Usage() if disqualifier == "incomplete-usage" else _observed()
    if disqualifier == "failed-finalize":
        rig.faults.arm("after")
    result: Final = await rig.finish(
        initial, wire, observed_usage=candidate,
        completed=disqualifier != "incomplete", cache_hit=disqualifier == "gateway-cache",
    )
    assert result.provenance != "observed_initial"
    if disqualifier == "failed-finalize":
        assert (await rig.finish(initial)).provenance != "observed_initial"
    state: Final = await rig.state(initial)
    assert not state.first_eligible
    if disqualifier == "count-unavailable":
        assert result.reason == "token_count_unavailable" and result.usage(5) is None
    rig.counter.unavailable = False
    if disqualifier in ("count-unavailable", "incomplete", "gateway-cache"):
        assert (await rig.run("following")).reason == "history_unavailable"
    rig.clock.advance(100000)
    restarted: Final = rig.fork()
    later_result: Final = await restarted.run("later", observed_usage=_observed())
    assert later_result.provenance != "observed_initial"
    assert later_result.comparison_id == state.comparison_id
    assert later_result.comparison_started_at == initial.reserved_at


async def test_bounded_completion_pruning_preserves_original_comparison_tombstone(rig: _Rig) -> None:
    first: Final = await rig.run("original", _native(_block(10)))
    for index in range(257):
        await rig.run(f"later-{index}", _native(_block(10)))
    original: Final = await rig.reserve("original")
    result: Final = await rig.finish(original, _native(_block(10)), observed_usage=_observed())
    assert result.provenance == "modeled"
    assert result.comparison_id == first.comparison_id


async def test_storage_deadline_bounds_all_conflict_attempts_and_preserves_cancellation(rig: _Rig) -> None:
    rig.faults.arm("conflict")
    rig.faults.before_write = lambda: asyncio.sleep(0.15)
    started: Final = time.monotonic()
    result: Final = await rig.estimator.reserve(rig.prepare("deadline"))
    assert result is not None and result.reason == "state_unavailable"
    assert time.monotonic() - started < 0.5
    assert rig.estimator.repairs
    cancelled: Final = asyncio.create_task(rig.estimator.reserve(rig.prepare("cancelled")))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
