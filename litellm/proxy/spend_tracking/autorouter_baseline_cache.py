from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import reduce
from types import MappingProxyType
from typing import Final, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from litellm.llms.anthropic.prompt_cache_prediction import (
    CountedBreakpoint,
    CountedPromptCachePlan,
    NativePredictionTarget,
    TokenCounter,
    UnsupportedCachePlan,
    count_cache_plan,
    count_prompt_tokens,
    parse_cache_plan,
    supported_baseline_recipient,
    supported_prediction_headers,
)
from litellm.proxy.db.autorouter_baseline_cache import BaselineStateFailure, BaselineStateSnapshot
from litellm.types.utils import CacheCreationTokenDetails, PromptTokensDetailsWrapper, Usage
from litellm.utils import get_prompt_cache_min_tokens

_MAX_TTL: Final = 3600
_MAX_REQUESTS: Final = 256
_MAX_VERSIONS: Final = 1024
_MAX_COUNTS: Final = 4096
_MAX_REPAIRS: Final = 1024
_COUNT_TIMEOUT: Final = 3.0
_STORE_TIMEOUT: Final = 0.25
_CAS_ATTEMPTS: Final = 4
_JSON_BODY: Final = TypeAdapter(dict[str, JsonValue])


class BaselineCacheEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[2] = 2
    comparison_id: str | None = None
    comparison_started_at: float | None = None
    provenance: Literal["observed_initial", "modeled"] | None = None
    observed_usage: Usage | None = None
    status: Literal["estimated", "unknown"]
    reason: str
    input_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_5m_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_1h_input_tokens: int | None = Field(default=None, ge=0)

    def usage(self, completion_tokens: int) -> Usage | None:
        if self.status == "estimated" and self.provenance == "observed_initial" and self.observed_usage is not None:
            return self.observed_usage.model_copy(deep=True)
        if (
            self.status != "estimated"
            or self.input_tokens is None
            or self.cache_read_input_tokens is None
            or self.cache_creation_5m_input_tokens is None
            or self.cache_creation_1h_input_tokens is None
        ):
            return None
        writes: Final = self.cache_creation_5m_input_tokens + self.cache_creation_1h_input_tokens
        total: Final = self.input_tokens + self.cache_read_input_tokens + writes
        return Usage(
            prompt_tokens=total,
            completion_tokens=completion_tokens,
            total_tokens=total + completion_tokens,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=self.input_tokens,
                cached_tokens=self.cache_read_input_tokens,
                cache_creation_tokens=writes,
                cache_write_tokens=writes,
                cache_creation_token_details=CacheCreationTokenDetails(
                    ephemeral_5m_input_tokens=self.cache_creation_5m_input_tokens,
                    ephemeral_1h_input_tokens=self.cache_creation_1h_input_tokens,
                ),
            ),
        )

    def metadata(self) -> dict[str, JsonValue]:  # mutable-ok: spend-log JSON serializers require a plain dictionary
        return _JSON_BODY.validate_python(
            self.model_dump(mode="json", exclude_none=True, exclude=MappingProxyType({"observed_usage": True}))
        )


def unknown_estimate(reason: str) -> BaselineCacheEstimate:
    return BaselineCacheEstimate(status="unknown", reason=reason)


def _observed_initial(usage: Usage | None) -> Usage | None:
    if usage is None or usage.prompt_tokens_details is None:
        return None
    details: Final = usage.prompt_tokens_details
    try:
        read: Final = details.cached_tokens
        writes: Final = details.cache_creation_tokens
        text: Final = details.text_tokens
        if read is None or writes is None or text is None or min(read, writes, text) < 0:
            return None
        if writes:
            creation: Final = details.cache_creation_token_details
            if (
                creation is None
                or creation.ephemeral_5m_input_tokens is None
                or creation.ephemeral_1h_input_tokens is None
                or min(creation.ephemeral_5m_input_tokens, creation.ephemeral_1h_input_tokens) < 0
                or creation.ephemeral_5m_input_tokens + creation.ephemeral_1h_input_tokens != writes
            ):
                return None
    except AttributeError:
        return None
    if (
        usage.prompt_tokens != text + read + writes
        or usage.prompt_tokens <= 0
        or usage.completion_tokens < 0
        or usage.total_tokens != usage.prompt_tokens + usage.completion_tokens
    ):
        return None
    return usage.model_copy(deep=True)


@dataclass(frozen=True, slots=True)
class BaselineReservation:
    scope: str
    request_id: str
    reserved_at: float
    target: NativePredictionTarget
    baseline_deployment_id: str


class _Pending(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    request_id: str
    reserved_at: float
    invalidated_reason: str | None = None


class _Completed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    request_id: str
    completed_at: float
    estimate: BaselineCacheEstimate
    terminal: bool = False


class _Version(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    fingerprint: str
    content_fingerprint: str
    prefix_tokens: int = Field(ge=0)
    ttl_seconds: Literal[300, 3600]
    started_at: float
    available_at: float
    expires_at: float
    uncertain: bool = False


class _State(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    comparison_id: str
    comparison_started_at: float
    first_request_id: str | None
    first_eligible: bool
    uncertain_before: float
    invalidated_before: float | None = None
    pending: tuple[_Pending, ...] = ()
    completed: tuple[_Completed, ...] = ()
    versions: tuple[_Version, ...] = ()


def _with_comparison(estimate: BaselineCacheEstimate, state: _State) -> BaselineCacheEstimate:
    return estimate.model_copy(
        update=MappingProxyType(
            {
                "comparison_id": state.comparison_id,
                "comparison_started_at": state.comparison_started_at,
                "provenance": estimate.provenance or ("modeled" if estimate.status == "estimated" else None),
            }
        )
    )


@dataclass(frozen=True, slots=True)
class _Repair:
    scope: str
    request_id: str
    reserved_at: float
    observed_at: float
    reason: str | None
    kind: Literal["active", "terminal", "cancel", "finish"]


@dataclass(frozen=True, slots=True)
class _Update:
    state: _State
    estimate: BaselineCacheEstimate | None


def _repair_state(state: _State, repair: _Repair) -> _State:
    prior: Final = next((item for item in state.completed if item.request_id == repair.request_id), None)
    pending: Final = next((item for item in state.pending if item.request_id == repair.request_id), None)
    active: Final = pending is not None and pending.invalidated_reason is not None
    reason: Final = (
        pending.invalidated_reason if repair.kind == "finish" and pending is not None and active else repair.reason
    )
    final: Final = (prior is not None and prior.terminal) or repair.kind in ("terminal", "cancel")
    terminal: Final = final or (repair.kind == "finish" and not active)
    remaining: Final = tuple(item for item in state.pending if item.request_id != repair.request_id)
    estimate: Final = unknown_estimate(reason or "request_cancelled")
    return _State(
        comparison_id=state.comparison_id,
        comparison_started_at=state.comparison_started_at,
        first_request_id=state.first_request_id,
        first_eligible=False,
        uncertain_before=max(state.uncertain_before, repair.observed_at) if reason else state.uncertain_before,
        invalidated_before=(
            max(state.invalidated_before or 0.0, repair.observed_at) if reason else state.invalidated_before
        ),
        pending=(
            remaining
            if terminal
            else (
                *remaining,
                _Pending(
                    request_id=repair.request_id,
                    reserved_at=pending.reserved_at if pending is not None else repair.reserved_at,
                    invalidated_reason=reason,
                ),
            )
        ),
        completed=(
            (
                *(item for item in state.completed if item.request_id != repair.request_id),
                _Completed(
                    request_id=repair.request_id,
                    completed_at=max(repair.observed_at, prior.completed_at if prior is not None else 0.0),
                    estimate=prior.estimate if prior is not None and prior.terminal and reason is None else estimate,
                    terminal=final,
                ),
            )
            if terminal
            else tuple(item for item in state.completed if item.request_id != repair.request_id)
        ),
        versions=state.versions,
    )


class BaselineStateStore(Protocol):
    async def read(self, scope: str, now: float) -> BaselineStateSnapshot | BaselineStateFailure: ...

    async def exchange(
        self, scope: str, before: BaselineStateSnapshot, serialized: str, now: float
    ) -> bool | BaselineStateFailure: ...


@dataclass(frozen=True, slots=True)
class _Count:
    tokens: int
    expires_at: float


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _key(caller: str, session: str, router: str, deployment: str, target: NativePredictionTarget) -> str:
    return "autorouter-baseline-cache:v2:" + _digest(
        (caller, session, router, deployment, target.model, target.api_key, target.api_base)
    )


def _bounded(state: _State, now: float) -> _State:
    stale_pending: Final = tuple(item for item in state.pending if item.reserved_at < now - 2 * _MAX_TTL)
    pending: Final = tuple(item for item in state.pending if item.reserved_at >= now - 2 * _MAX_TTL)
    needed_since: Final = min((now - _MAX_TTL, *(item.reserved_at for item in pending)))
    live_versions: Final = tuple(
        version
        for version in state.versions
        if version.expires_at >= needed_since
        and (state.invalidated_before is None or version.available_at > state.invalidated_before)
    )
    dropped_versions: Final = live_versions[:-_MAX_VERSIONS]
    dropped_pending: Final = pending[:-_MAX_REQUESTS]
    uncertainty: Final = max(
        (
            state.uncertain_before,
            *(version.started_at for version in dropped_versions),
            now if stale_pending or dropped_pending else 0.0,
        )
    )
    return _State(
        comparison_id=state.comparison_id,
        comparison_started_at=state.comparison_started_at,
        first_request_id=state.first_request_id,
        first_eligible=state.first_eligible and not stale_pending and not dropped_pending,
        uncertain_before=uncertainty,
        invalidated_before=state.invalidated_before,
        pending=pending[-_MAX_REQUESTS:],
        completed=state.completed[-_MAX_REQUESTS:],
        versions=live_versions[-_MAX_VERSIONS:],
    )


def _eligible(plan: CountedPromptCachePlan, minimum: int) -> tuple[CountedBreakpoint, ...]:
    return tuple(marker for marker in plan.breakpoints if marker.prefix_tokens >= minimum)


def _matching(state: _State, markers: tuple[CountedBreakpoint, ...], started: float) -> tuple[_Version, ...]:
    return tuple(
        version
        for version in state.versions
        if not version.uncertain
        and version.available_at <= started < version.expires_at
        and any(
            version.fingerprint in marker.lookback_fingerprints
            and version.prefix_tokens <= marker.prefix_tokens
            and version.ttl_seconds == marker.ttl_seconds
            for marker in markers
        )
    )


def _ambiguous(state: _State, markers: tuple[CountedBreakpoint, ...], started: float) -> tuple[_Version, ...]:
    return tuple(
        version
        for version in state.versions
        if version.available_at <= started < version.expires_at
        and any(
            version.content_fingerprint in marker.lookback_content_fingerprints
            and (version.uncertain or version.ttl_seconds != marker.ttl_seconds)
            for marker in markers
        )
    )


def _estimate(
    state: _State, request_id: str, plan: CountedPromptCachePlan, minimum: int, started: float
) -> BaselineCacheEstimate:
    markers: Final = _eligible(plan, minimum)
    if not markers:
        return BaselineCacheEstimate(
            status="estimated",
            reason="below_cache_minimum" if plan.breakpoints else "no_cache_breakpoints",
            input_tokens=plan.total_tokens,
            cache_read_input_tokens=0,
            cache_creation_5m_input_tokens=0,
            cache_creation_1h_input_tokens=0,
        )
    if any(item.request_id != request_id and item.reserved_at <= started for item in state.pending):
        return unknown_estimate("pending_request")
    if _ambiguous(state, markers, started):
        return unknown_estimate("cache_ttl_changed")
    candidates: Final = _matching(state, markers, started)
    read: Final = max((version.prefix_tokens for version in candidates), default=0)
    end: Final = markers[-1].prefix_tokens
    if read < end and started < state.uncertain_before + max(marker.ttl_seconds for marker in markers):
        return unknown_estimate("history_unavailable")
    one_hour: Final = max(
        (marker.prefix_tokens for marker in markers if marker.ttl_seconds == 3600 and marker.prefix_tokens > read),
        default=read,
    )
    expired: Final = any(
        version.expires_at <= started and any(version.fingerprint in marker.lookback_fingerprints for marker in markers)
        for version in state.versions
    )
    return BaselineCacheEstimate(
        status="estimated",
        reason="cache_prefix_available" if read else "cache_prefix_expired" if expired else "cache_prefix_cold",
        input_tokens=plan.total_tokens - end,
        cache_read_input_tokens=read,
        cache_creation_5m_input_tokens=end - one_hour,
        cache_creation_1h_input_tokens=one_hour - read,
    )


def _new_versions(
    state: _State,
    markers: tuple[CountedBreakpoint, ...],
    started: float,
    available: float,
) -> tuple[_Version, ...]:
    ambiguous: Final = _ambiguous(state, markers, started)
    longest_ttl: Final = max((version.expires_at - version.started_at for version in ambiguous), default=0)
    candidates: Final = () if ambiguous else _matching(state, markers, started)
    hit: Final = max(candidates, key=lambda version: version.prefix_tokens, default=None)
    refresh: Final = (
        (
            _Version(
                fingerprint=hit.fingerprint,
                content_fingerprint=hit.content_fingerprint,
                prefix_tokens=hit.prefix_tokens,
                ttl_seconds=hit.ttl_seconds,
                started_at=started,
                available_at=available,
                expires_at=started + hit.ttl_seconds,
            ),
        )
        if hit is not None and all(marker.fingerprint != hit.fingerprint for marker in markers)
        else ()
    )
    return (
        *refresh,
        *(
            _Version(
                fingerprint=marker.fingerprint,
                content_fingerprint=marker.content_fingerprint,
                prefix_tokens=marker.prefix_tokens,
                ttl_seconds=3600 if marker.ttl_seconds == 3600 else 300,
                started_at=started,
                available_at=available,
                expires_at=started + max(longest_ttl, marker.ttl_seconds),
                uncertain=bool(ambiguous),
            )
            for marker in markers
        ),
    )


class BaselineCacheEstimator:
    def __init__(
        self,
        store: BaselineStateStore,
        clock: Callable[[], float] = time.time,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.store = store
        self.clock = clock
        self.token_counter = token_counter
        self.counts: Mapping[str, _Count] = MappingProxyType({})
        self.count_slots = asyncio.Semaphore(8)
        self.repairs: Mapping[tuple[str, str], _Repair] = MappingProxyType({})
        self.uncertainty_floor = 0.0

    def prepare(
        self,
        *,
        caller_key_hash: str,
        session_id: str,
        router_id: str,
        baseline_deployment_id: str,
        target: NativePredictionTarget,
        request_id: str,
    ) -> BaselineReservation | BaselineCacheEstimate:
        if not all((caller_key_hash, session_id, router_id, baseline_deployment_id, request_id)):
            return unknown_estimate("missing_baseline_scope")
        try:
            now: Final = self.clock()
        except Exception:  # noqa: BLE001  # optional preparation cannot prevent generation
            return unknown_estimate("estimator_unavailable")
        if not math.isfinite(now):
            return unknown_estimate("estimator_unavailable")
        return BaselineReservation(
            _key(caller_key_hash, session_id, router_id, baseline_deployment_id, target),
            _digest(request_id),
            now,
            target,
            baseline_deployment_id,
        )

    def _repair_time(self, reservation: BaselineReservation) -> float:
        try:
            now: Final = self.clock()
            if math.isfinite(now):
                return max(now, reservation.reserved_at)
        except Exception:  # noqa: BLE001  # emergency cleanup must survive the original clock fault
            return max(time.time(), reservation.reserved_at)
        return max(time.time(), reservation.reserved_at)

    def _defer(
        self,
        reservation: BaselineReservation,
        reason: str | None,
        kind: Literal["active", "terminal", "cancel", "finish"],
    ) -> None:
        now: Final = self._repair_time(reservation)
        key: Final = (reservation.scope, reservation.request_id)
        prior: Final = self.repairs.get(key)
        repair: Final = _Repair(
            scope=reservation.scope,
            request_id=reservation.request_id,
            reserved_at=reservation.reserved_at,
            observed_at=max(now, prior.observed_at if prior is not None else 0.0),
            reason=prior.reason if kind == "finish" and prior is not None else reason,
            kind=(
                prior.kind
                if kind == "finish" and prior is not None
                else "terminal"
                if kind == "active" and prior is not None and prior.kind in ("terminal", "cancel")
                else kind
            ),
        )
        retained: Final = tuple(
            (identity, item)
            for identity, item in self.repairs.items()
            if identity != key and max(item.reserved_at + 2 * _MAX_TTL, item.observed_at + _MAX_TTL) >= now
        )
        self.uncertainty_floor = max(
            (
                self.uncertainty_floor,
                *(item.observed_at for _, item in retained[: -(_MAX_REPAIRS - 1)] if item.reason is not None),
            )
        )
        self.repairs = MappingProxyType(
            {identity: item for identity, item in (*retained[-(_MAX_REPAIRS - 1) :], (key, repair))}
        )

    def defer(self, reservation: BaselineReservation, reason: str, *, completed: bool) -> None:
        self._defer(reservation, reason, "terminal" if completed else "active")

    def defer_cancel(self, reservation: BaselineReservation) -> None:
        self._defer(reservation, None, "cancel")

    async def _transition(
        self,
        reservation: BaselineReservation,
        operation: Callable[[_State, bool, float], _Update],
        *,
        admit: bool = False,
    ) -> BaselineCacheEstimate | BaselineStateFailure | None:
        async def attempt(remaining: int) -> BaselineCacheEstimate | BaselineStateFailure | None:
            now: Final = self.clock()
            repairs: Final = tuple(item for item in self.repairs.values() if item.scope == reservation.scope)
            floor: Final = self.uncertainty_floor
            before: Final = await self.store.read(reservation.scope, now)
            if isinstance(before, BaselineStateFailure):
                return before
            try:
                loaded: Final = (
                    _State.model_validate_json(before.state)
                    if before.state is not None
                    else _State(
                        comparison_id=_digest((reservation.scope, reservation.request_id, reservation.reserved_at)),
                        comparison_started_at=now,
                        first_request_id=reservation.request_id if admit else None,
                        first_eligible=admit,
                        uncertain_before=now,
                    )
                )
            except ValidationError:
                return BaselineStateFailure("state_corrupt")
            initial: Final = loaded.model_copy(
                update=MappingProxyType(
                    {
                        "first_eligible": loaded.first_eligible
                        and (
                            floor == 0.0 or (loaded.comparison_started_at > floor and reservation.reserved_at > floor)
                        ),
                        "uncertain_before": max(loaded.uncertain_before, floor),
                        "invalidated_before": (
                            max(loaded.invalidated_before or 0.0, floor) if floor > 0.0 else loaded.invalidated_before
                        ),
                    }
                )
            )
            state: Final = _bounded(reduce(_repair_state, repairs, initial), now)
            update: Final = operation(state, before.state is None, now)
            after: Final = _bounded(update.state, now)
            exchanged: Final = await self.store.exchange(reservation.scope, before, after.model_dump_json(), now)
            if isinstance(exchanged, BaselineStateFailure):
                return exchanged
            if not exchanged:
                return await attempt(remaining - 1) if remaining > 1 else BaselineStateFailure("state_contention")
            acknowledged: Final = MappingProxyType({(item.scope, item.request_id): item for item in repairs})
            self.repairs = MappingProxyType(
                {identity: item for identity, item in self.repairs.items() if acknowledged.get(identity) is not item}
            )
            return _with_comparison(update.estimate, after) if update.estimate is not None else None

        try:
            return await asyncio.wait_for(attempt(_CAS_ATTEMPTS), timeout=_STORE_TIMEOUT)
        except TimeoutError:
            return BaselineStateFailure()

    async def reserve(self, reservation: BaselineReservation) -> BaselineCacheEstimate | None:
        def register(state: _State, _missing: bool, _now: float) -> _Update:
            finished: Final = next(
                (item for item in state.completed if item.request_id == reservation.request_id), None
            )
            if finished is not None:
                return _Update(state, finished.estimate)
            pending: Final = next((item for item in state.pending if item.request_id == reservation.request_id), None)
            if pending is not None:
                return _Update(
                    state, unknown_estimate(pending.invalidated_reason) if pending.invalidated_reason else None
                )
            return _Update(
                state.model_copy(
                    update=MappingProxyType(
                        {
                            "first_eligible": state.first_eligible and state.first_request_id == reservation.request_id,
                            "pending": (
                                *state.pending,
                                _Pending(request_id=reservation.request_id, reserved_at=reservation.reserved_at),
                            ),
                        }
                    )
                ),
                None,
            )

        result: Final = await self._transition(reservation, register, admit=True)
        if isinstance(result, BaselineStateFailure):
            self.defer(reservation, result.reason, completed=False)
            return unknown_estimate(result.reason)
        return result

    async def _count(self, target: NativePredictionTarget, body: Mapping[str, JsonValue]) -> int | None:
        cache_key: Final = _digest((target.model, target.api_key, target.api_base, _JSON_BODY.validate_python(body)))
        now: Final = self.clock()
        cached: Final = self.counts.get(cache_key)
        if cached is not None and cached.expires_at > now:
            return cached.tokens

        async def execute() -> int | None:
            async with self.count_slots:
                return (
                    await self.token_counter(target.model, target.api_key, body)
                    if self.token_counter is not None
                    else await count_prompt_tokens(target.model, target.api_key, body, api_base=target.api_base)
                )

        try:
            tokens: Final = await asyncio.wait_for(execute(), timeout=_COUNT_TIMEOUT)
        except Exception:  # noqa: BLE001  # provider counting cannot fail a completed generation
            return None
        if tokens is None or tokens < 0:
            return None
        retained: Final = tuple(
            (key, value) for key, value in self.counts.items() if value.expires_at > now and key != cache_key
        )[-(_MAX_COUNTS - 1) :]
        self.counts = MappingProxyType(
            {key: value for key, value in (*retained, (cache_key, _Count(tokens, now + _MAX_TTL)))}
        )
        return tokens

    async def finalize(
        self,
        reservation: BaselineReservation,
        *,
        wire: httpx.Request,
        request_started_at: float,
        available_at: float,
        completed: bool = True,
        cache_hit: bool = False,
        observed_cache_tokens: int = 0,
        observed_usage: Usage | None = None,
    ) -> BaselineCacheEstimate:
        if cache_hit:
            await self.cancel(reservation)
            return unknown_estimate("response_cache_hit")
        if not completed:
            return await self.invalidate(reservation, "incomplete_response")
        if not reservation.reserved_at <= request_started_at <= available_at <= self.clock():
            return await self._finish(reservation, None, request_started_at, available_at, "invalid_request_timing")
        if not supported_baseline_recipient(reservation.target, wire):
            return await self._finish(
                reservation, None, request_started_at, available_at, "unsupported_baseline_recipient"
            )
        try:
            body: Final = _JSON_BODY.validate_json(wire.content)
        except (ValidationError, RuntimeError, httpx.RequestNotRead):
            return await self._finish(reservation, None, request_started_at, available_at, "invalid_wire_request")
        initial: Final = _observed_initial(observed_usage) if body.get("model") == reservation.target.model else None
        if not supported_prediction_headers(wire.headers):
            return await self._finish(
                reservation, None, request_started_at, available_at, "unsupported_request_headers", initial
            )
        plan: Final = parse_cache_plan(body)
        if isinstance(plan, UnsupportedCachePlan):
            return await self._finish(reservation, None, request_started_at, available_at, plan.reason, initial)
        if not plan.breakpoints and observed_cache_tokens > 0:
            return await self._finish(
                reservation, None, request_started_at, available_at, "implicit_cache_without_breakpoints", initial
            )

        async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
            return await self._count(reservation.target, body)

        try:
            counted: Final = await asyncio.wait_for(
                count_cache_plan(reservation.target.model, reservation.target.api_key, plan, token_counter=count),
                timeout=_COUNT_TIMEOUT,
            )
        except TimeoutError:
            return await self._finish(
                reservation, None, request_started_at, available_at, "token_count_timeout", initial
            )
        return await self._finish(
            reservation,
            None if isinstance(counted, UnsupportedCachePlan) else counted,
            request_started_at,
            available_at,
            counted.reason if isinstance(counted, UnsupportedCachePlan) else None,
            initial,
        )

    async def _finish(
        self,
        reservation: BaselineReservation,
        plan: CountedPromptCachePlan | None,
        started: float,
        available: float,
        reason: str | None,
        observed_usage: Usage | None = None,
    ) -> BaselineCacheEstimate:
        minimum: Final = get_prompt_cache_min_tokens(reservation.target.model)

        def finish(state: _State, missing: bool, now: float) -> _Update:
            prior: Final = next((item for item in state.completed if item.request_id == reservation.request_id), None)
            if prior is not None:
                return _Update(state, prior.estimate)
            pending: Final = next((item for item in state.pending if item.request_id == reservation.request_id), None)
            if pending is not None and pending.invalidated_reason:
                return _Update(state, unknown_estimate(pending.invalidated_reason))
            initial: Final = (
                observed_usage is not None
                and pending is not None
                and state.first_eligible
                and state.first_request_id == reservation.request_id
            )
            unavailable: Final = (
                "history_unavailable" if missing else "reservation_unavailable" if pending is None else reason
            )
            estimate: Final = _with_comparison(
                BaselineCacheEstimate(
                    status="estimated",
                    reason="observed_initial_baseline",
                    provenance="observed_initial",
                    observed_usage=observed_usage,
                )
                if initial
                else _estimate(state, reservation.request_id, plan, minimum, started)
                if plan is not None and pending is not None
                else unknown_estimate(unavailable or "unsupported_request"),
                state,
            )
            history: Final = (
                state
                if plan is not None and pending is not None
                else _repair_state(
                    state,
                    _Repair(
                        reservation.scope,
                        reservation.request_id,
                        reservation.reserved_at,
                        now,
                        unavailable or "unsupported_request",
                        "finish",
                    ),
                )
            )
            versions: Final = (
                _new_versions(state, _eligible(plan, minimum), started, available)
                if plan is not None and pending is not None
                else ()
            )
            return _Update(
                history.model_copy(
                    update=MappingProxyType(
                        {
                            "first_eligible": False,
                            "pending": tuple(
                                item for item in history.pending if item.request_id != reservation.request_id
                            ),
                            "completed": (
                                *(item for item in history.completed if item.request_id != reservation.request_id),
                                _Completed(request_id=reservation.request_id, completed_at=now, estimate=estimate),
                            ),
                            "versions": (*history.versions, *versions),
                        }
                    )
                ),
                estimate,
            )

        result: Final = await self._transition(reservation, finish)
        if isinstance(result, BaselineStateFailure):
            self._defer(reservation, result.reason, "finish")
            return unknown_estimate(result.reason)
        return result if result is not None else unknown_estimate("estimator_unavailable")

    async def invalidate(
        self, reservation: BaselineReservation, reason: str, *, completed: bool = True
    ) -> BaselineCacheEstimate:
        self.defer(reservation, reason, completed=completed)

        def invalidated(state: _State, _missing: bool, _now: float) -> _Update:
            return _Update(state, unknown_estimate(reason))

        result: Final = await self._transition(reservation, invalidated)
        if isinstance(result, BaselineStateFailure):
            return unknown_estimate(result.reason)
        return result if result is not None else unknown_estimate(reason)

    async def cancel(self, reservation: BaselineReservation) -> None:
        self.defer_cancel(reservation)

        def cancelled(state: _State, _missing: bool, _now: float) -> _Update:
            return _Update(state, None)

        await self._transition(reservation, cancelled)
