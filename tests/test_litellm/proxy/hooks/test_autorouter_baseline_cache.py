import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Generator, Mapping
from contextlib import contextmanager
from datetime import datetime
from types import MappingProxyType
from typing import (
    Final,
    Literal,
    cast,  # noqa: TID251  # runtime-checked iterator items are safely widened to object
)

import httpx
import pytest
import respx
from fastapi import HTTPException
from pydantic import JsonValue, TypeAdapter
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.prompt_cache_prediction import NativePredictionTarget, TokenCounter
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.autorouter_baseline_cache import (
    AutoRouterBaselineCache,
    BaselineCacheContext,
    cancel_baseline_cache,
    finalize_baseline_cache,
    invalidate_baseline_cache,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
from litellm.proxy.spend_tracking.autorouter_baseline_cache import (
    BaselineCacheEstimate,
    BaselineCacheEstimator,
    BaselineReservation,
    unknown_estimate,
)
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.router import RetryPolicy
from litellm.types.utils import CallTypes, ModelResponse, StandardLoggingRoutingDecision
from tests.test_litellm.proxy._baseline_cache_test_helpers import InMemoryBaselineStore

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_OBJECTS: Final = TypeAdapter(dict[str, object])
_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])
_MESSAGES_JSON: Final = """[{"role":"user","content":[
    {"type":"text","text":"stable","cache_control":{"type":"ephemeral","ttl":"1h"}},
    {"type":"text","text":"question"}]}]"""
_THINKING_MESSAGES: Final = """[
    {"role":"user","content":[{"type":"text","text":"stable","cache_control":{"type":"ephemeral","ttl":"1h"}}]},
    {"role":"assistant","content":[{"type":"thinking","thinking":"reasoning","signature":"invalid"},{"type":"text","text":"answer"}]},
    {"role":"user","content":"question"}]"""
_MODELS: Final = _MESSAGES.validate_json("""[
    {"model_name":"test-router","litellm_params":{"model":"auto_router/complexity_router",
      "complexity_router_config":{"tiers":{"SIMPLE":"sonnet","MEDIUM":"sonnet","COMPLEX":"sonnet",
      "REASONING":"opus"},"session_affinity":false,
      "keyword_tier_rules":[{"keywords":["USE_OPUS"],"tier":"REASONING"}]}}},
    {"model_name":"sonnet","litellm_params":{"model":"anthropic/claude-sonnet-5","api_key":"test-selected"},
      "model_info":{"id":"selected"}},
    {"model_name":"opus","litellm_params":{"model":"anthropic/claude-opus-5","api_key":"test-selected"},
      "model_info":{"id":"baseline"}}]""")


def _message(completed: bool, model: str) -> Mapping[str, JsonValue]:
    return _JSON_OBJECT.validate_json(f"""{{
        "id":"msg_baseline_test","type":"message","role":"assistant","model":{json.dumps(model)},
        "content":{'[{"type":"text","text":"OK"}]' if completed else "[]"},
        "stop_reason":{'"end_turn"' if completed else "null"},"stop_sequence":null,
        "usage":{{"input_tokens":1000,"output_tokens":{10 if completed else 0},
          "cache_creation_input_tokens":5000,"cache_read_input_tokens":0,
          "cache_creation":{{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":5000}}}}}}""")


_EVENTS: Final = _MESSAGES.validate_json("""[
    {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}},
    {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"OK"}},
    {"type":"content_block_stop","index":0},
    {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}},
    {"type":"message_stop"}
]""")
pytestmark: Final = pytest.mark.asyncio


def _is_unknown(estimate: BaselineCacheEstimate | None, reason: str) -> bool:
    return estimate is not None and estimate.model_copy(
        update=MappingProxyType({"comparison_id": None, "comparison_started_at": None})
    ) == unknown_estimate(reason)


class _Capture(CustomLogger):
    def __init__(self) -> None:
        self.payloads: asyncio.Queue[Mapping[str, object]] = asyncio.Queue()
        self.terminal_errors: asyncio.Queue[Exception] = asyncio.Queue()
        self.attempts: tuple[Logging, ...] = ()
        self.replacement_error = HTTPException(status_code=429, detail="transformed terminal provider error")

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        if kwargs.get("litellm_call_id") in ("first", "following", "matching"):
            self.payloads.put_nowait(_OBJECTS.validate_python(kwargs.get("standard_logging_object")))

    async def async_pre_call_deployment_hook(self, kwargs: Mapping[str, object], call_type: CallTypes | None) -> None:
        logging_obj: Final = kwargs.get("litellm_logging_obj")
        if isinstance(logging_obj, Logging) and call_type == CallTypes.anthropic_messages:
            self.attempts = (*self.attempts, logging_obj)

    async def async_post_call_failure_hook(
        self,
        request_data: Mapping[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> HTTPException:
        assert "litellm_logging_obj" not in request_data
        self.terminal_errors.put_nowait(original_exception)
        return self.replacement_error

    async def payload(self) -> Mapping[str, object]:
        return await asyncio.wait_for(self.payloads.get(), timeout=20)


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now
        self.failures_remaining = 0
        self.failure_at: int | None = None
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if self.calls == self.failure_at or self.failures_remaining:
            self.failures_remaining = max(0, self.failures_remaining - 1)
            raise ValueError("injected estimator clock failure")
        return self.now


async def _count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
    assert model == "claude-opus-5"
    return 6000 if "question" in json.dumps(_JSON_OBJECT.validate_python(body)) else 5000


class _CountingGate:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> None:
        self.started.set()
        await self.release.wait()

    async def count(self, model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
        await self.wait()
        return await _count(model, api_key, body)


class _Rig:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        now: float | None = None,
        retries: int = 0,
        lookup_fails: bool = False,
        count: TokenCounter = _count,
    ) -> None:
        self.clock: Final = _Clock(round(time.time(), 6) if now is None else now)
        self.store: Final = InMemoryBaselineStore()
        self.estimator: Final = BaselineCacheEstimator(
            self.store, token_counter=count, clock=time.time if now is None else self.clock
        )
        self.router: Final = Router(
            model_list=_MODELS,
            num_retries=retries,
            retry_policy=RetryPolicy(RateLimitErrorRetries=retries),
            disable_cooldowns=True,
        )

        def get_router() -> Router:
            if lookup_fails:
                raise ValueError("injected baseline deployment lookup failure")
            return self.router

        self.hook: Final = AutoRouterBaselineCache(None, router=get_router, estimator=self.estimator)
        self.capture: Final = _Capture()
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        for name in ("ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(
            litellm,
            "callbacks",
            [self.hook, self.capture],  # mutable-ok: LiteLLM mutates callback registries
        )
        for name in ("success_callback", "failure_callback"):
            monkeypatch.setattr(litellm, name, [])  # mutable-ok: LiteLLM mutates callback registries
        for name in ("_async_success_callback", "_async_failure_callback"):
            monkeypatch.setattr(litellm, name, [self.capture])  # mutable-ok: LiteLLM mutates callback registries

    def logging(self, request_id: str = "first", stream: bool = False) -> Logging:
        return Logging(  # pyright: ignore[reportUnknownMemberType]  # production constructor has legacy argument types
            model="anthropic/claude-sonnet-5",
            messages=_MESSAGES.validate_json(_MESSAGES_JSON),
            stream=stream,
            call_type=CallTypes.anthropic_messages.value,
            start_time=datetime.now(),  # noqa: DTZ005  # native Logging requires naive timestamps
            litellm_call_id=request_id,
            function_id=request_id,
            kwargs=_OBJECTS.validate_json('{"litellm_session_id":"baseline-session"}'),
        )

    async def reserve(self, request_id: str) -> BaselineReservation:
        reservation: Final = self.estimator.prepare(
            caller_key_hash="test-caller-hash",
            session_id="baseline-session",
            router_id="test-router",
            baseline_deployment_id="baseline",
            request_id=request_id,
            target=NativePredictionTarget("claude-opus-5", "test-selected", "https://api.anthropic.com"),
        )
        assert isinstance(reservation, BaselineReservation)
        assert await self.estimator.reserve(reservation) is None
        return reservation

    def stamp(self, logging_obj: Logging, *, complete: bool = True) -> None:
        timestamp: Final = datetime.fromtimestamp(self.clock.now)  # noqa: DTZ006  # native Logging timing is naive
        logging_obj.completion_start_time = timestamp  # rebind-ok: supply native provider timing evidence
        logging_obj.model_call_details.update(  # pyright: ignore[reportUnknownMemberType]  # legacy native evidence dictionary
            httpx_response=_upstream(_wire(_stream(logging_obj))),
            api_call_start_time=timestamp,
            completion_start_time=timestamp,
            custom_llm_provider="anthropic",
            response_cost=0.125,
            stream=_stream(logging_obj),
            prompt_cache_response_complete=complete,
        )

    async def native(self, request_id: str = "first", *, stream: bool = False) -> Logging:
        logging_obj: Final = self.logging(request_id, stream)
        logging_obj.baseline_cache_context = BaselineCacheContext(self.estimator, await self.reserve(request_id))
        self.stamp(logging_obj)
        return logging_obj

    async def finish(self, request_id: str = "following") -> BaselineCacheEstimate | None:
        logging_obj: Final = await self.native(request_id)
        await finalize_baseline_cache(logging_obj, ModelResponse())
        return logging_obj.baseline_cache_estimate

    async def recover(self) -> None:
        boundary: Final = max((self.clock.now, *(repair.observed_at for repair in self.estimator.repairs.values())))
        assert boundary - self.clock.now < 60
        self.clock.now = round(boundary + 1, 6)
        assert _is_unknown(await self.finish(), "history_unavailable")
        self.clock.now += 1
        estimate: Final = await self.finish("matching")
        assert estimate is not None and estimate.cache_read_input_tokens == 5000


class _CallContext(TypedDict):
    litellm_logging_obj: NotRequired[ReadOnly[Logging]]
    litellm_call_id: ReadOnly[str]
    litellm_metadata: ReadOnly[Mapping[str, object]]
    litellm_session_id: ReadOnly[str]


def _kwargs(logging_obj: Logging, trusted: bool = True, *, explicit_logging: bool = True) -> _CallContext:
    context: Final = _OBJECTS.validate_json('{"litellm_metadata":{"user_api_key_hash":"test-caller-hash"}}')
    Router._record_routing_decision(  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]  # production trusted stamp owner
        context,
        StandardLoggingRoutingDecision(
            router_model_name="test-router",
            router_type="complexity",
            routed_model="sonnet",
            cause="heuristic_scorer",
            conversation_continuing=True,
            savings_baseline_model="anthropic/claude-opus-5",
            savings_baseline_deployment_id="baseline",
        ),
    )
    metadata: Final = _OBJECTS.validate_python(context["litellm_metadata"])
    if not trusted:
        metadata["_autorouter_baseline_route"] = _JSON_OBJECT.validate_json(
            '{"router_name":"test-router","baseline_model":"anthropic/claude-opus-5","baseline_deployment_id":"baseline"}'
        )
    envelope: Final[_CallContext] = {
        "litellm_call_id": logging_obj.litellm_call_id,
        "litellm_session_id": "baseline-session",
        "litellm_metadata": metadata,
    }
    supplied: Final[_CallContext] = {**envelope, "litellm_logging_obj": logging_obj}
    return supplied if explicit_logging else envelope


def _stream(logging_obj: Logging) -> bool:
    return logging_obj.stream is True  # pyright: ignore[reportUnknownMemberType]  # normalize the legacy Logging flag


def _sse(completed: bool = True, model: str = "claude-sonnet-5") -> tuple[bytes, ...]:
    events: Final = (
        {  # mutable-ok: json.dumps needs a concrete event dictionary
            "type": "message_start",
            "message": _message(False, model),
        },
        *_EVENTS,
    )
    return tuple(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
        for event in (events if completed else events[:-1])
    )


def _upstream(request: httpx.Request) -> httpx.Response:
    body: Final = _JSON_OBJECT.validate_json(request.content)
    model: Final = body.get("model")
    assert isinstance(model, str)
    stream: Final = body.get("stream") is True
    content: Final = b"".join(_sse(model=model)) if stream else json.dumps(_message(True, model)).encode()
    return httpx.Response(200, content=content, request=request,
        headers=MappingProxyType({"content-type": "text/event-stream" if stream else "application/json"}),
    )


def _error(request: httpx.Request, code: int, message: str) -> httpx.Response:
    return httpx.Response(
        code,
        text='{"type":"error","error":{"type":"rate_limit_error","message":' + json.dumps(message) + "}}",
        headers=MappingProxyType({"retry-after": "0"}),
        request=request,
    )


@contextmanager
def _transport(upstream: Callable[[httpx.Request], httpx.Response]) -> Generator[respx.Route]:
    with respx.mock() as transport:
        yield transport.post("https://api.anthropic.com/v1/messages").mock(side_effect=upstream)


class _NativeOptions(TypedDict):
    api_key: NotRequired[ReadOnly[str]]
    num_retries: NotRequired[ReadOnly[int]]


async def _call(
    target: Router | None,
    logging_obj: Logging,
    *,
    trusted: bool = True,
    messages: str = _MESSAGES_JSON,
    explicit_logging: bool = True,
) -> None:
    invoke: Final = target.anthropic_messages if target else litellm.anthropic_messages  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # legacy native call signatures
    options: Final = _NativeOptions() if target else _NativeOptions(api_key="test-selected", num_retries=0)
    response: Final[object] = await invoke(  # pyright: ignore[reportUnknownVariableType]  # native Router returns an opaque SDK result
        model="test-router" if target else "anthropic/claude-sonnet-5",
        max_tokens=16,
        stream=_stream(logging_obj),
        messages=_MESSAGES.validate_json(messages),
        **options,
        **_kwargs(logging_obj, trusted, explicit_logging=explicit_logging),
    )
    assert response is not None
    if _stream(logging_obj):
        assert isinstance(response, AsyncIterator)
        stream: Final = cast(AsyncIterator[object], response)  # cast-ok: iterator checked; all items satisfy object
        assert tuple([chunk async for chunk in stream])


def _wire(stream: bool = False) -> httpx.Request:
    return httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers=MappingProxyType({"x-api-key": "test-selected"}),
        content=f'{{"model":"claude-sonnet-5","stream":{json.dumps(stream)},"messages":{_MESSAGES_JSON}}}',
    )


@pytest.mark.parametrize("stream", (False, True))
async def test_initial_baseline_turn_preserves_nonzero_cost_and_records_zero_savings(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    rig: Final = _Rig(monkeypatch)
    with _transport(_upstream):
        await _call(
            rig.router, rig.logging(stream=stream), messages=_MESSAGES_JSON.replace('"question"', '"USE_OPUS question"')
        )
        payload: Final = await rig.capture.payload()
    estimate: Final = _OBJECTS.validate_python(payload["autorouter_savings_estimate"])
    cost: Final = payload["response_cost"]
    assert isinstance(cost, float) and cost > 0
    assert payload["autorouter_savings"] == 0.0, estimate
    assert (estimate["version"], estimate["status"], estimate["provenance"]) == (2, "estimated", "observed_initial")
    assert estimate["comparison_id"] and estimate["comparison_started_at"]


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("trusted_stamp", (True, False))
async def test_native_dispatch_reserves_before_upstream_and_stamps_before_callbacks(
    monkeypatch: pytest.MonkeyPatch, stream: bool, trusted_stamp: bool
) -> None:
    rig: Final = _Rig(monkeypatch)
    with _transport(_upstream):
        for request_id in ("first", "following"):
            await _call(None, rig.logging(request_id, stream), trusted=trusted_stamp, explicit_logging=False)
            payload = await rig.capture.payload()
            estimate = _OBJECTS.validate_python(payload["autorouter_savings_estimate"])
            if not trusted_stamp or request_id == "first":
                assert estimate["status"] == "unknown" and payload["autorouter_savings"] is None
                if trusted_stamp:
                    assert estimate["reason"] == "history_unavailable"
            else:
                assert estimate["status"] == "estimated" and estimate["cache_read_input_tokens"] == 5000
                assert estimate["cache_creation_1h_input_tokens"] == 0
                saving = payload["autorouter_savings"]
                assert isinstance(saving, float) and saving < 0


@pytest.mark.parametrize(
    "stream,thinking,invalidation_fails",
    ((False, False, False), (True, False, False), (False, True, False), (False, True, True)),
)
async def test_native_router_retry_with_shared_logging_is_unknown_and_cannot_warm_baseline(
    monkeypatch: pytest.MonkeyPatch, stream: bool, thinking: bool, invalidation_fails: bool
) -> None:
    rig: Final = _Rig(monkeypatch, 1000.0 if thinking else None, retries=1)

    def upstream(request: httpx.Request) -> httpx.Response:
        if route.call_count:
            return _upstream(request)
        rig.clock.failures_remaining = 2 if invalidation_fails else 0
        if thinking:
            assert b'"signature": "invalid"' in request.content
            return _error(request, 400, "messages.1.content.0: Invalid `signature` in `thinking` block")
        return _error(request, 429, "retry")

    shared: Final = rig.logging(stream=stream)
    with _transport(upstream) as route:
        await _call(rig.router, shared, messages=_THINKING_MESSAGES if thinking else _MESSAGES_JSON)
        payload: Final = await rig.capture.payload()
        assert route.call_count == 2 and rig.capture.attempts == ((shared,) if thinking else (shared, shared))
        assert payload["autorouter_savings"] is None and shared.baseline_cache_context is None
        assert _OBJECTS.validate_python(payload["autorouter_savings_estimate"])["reason"] == (
            "estimator_unavailable" if invalidation_fails else "retried_request"
        )
        if thinking:
            assert b'"signature"' not in route.calls.last.request.content
        else:
            await _call(rig.router, rig.logging("following", stream))
            following: Final = await rig.capture.payload()
            assert route.call_count == 3 and following["autorouter_savings"] is None
            assert _OBJECTS.validate_python(following["autorouter_savings_estimate"])["reason"] == "history_unavailable"


@pytest.mark.parametrize(
    "operation,retry_owns_reservation,fault",
    (
        *(("finalize", retry, fault) for retry in (False, True) for fault in ("exception", "before", "after")),
        ("finalize", True, "none"),
        ("cancel", False, "none"),
        ("cancel", False, "exception"),
    ),
)
async def test_late_finalize_failure_cleans_original_reservation_without_overwriting_replacement(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    retry_owns_reservation: bool,
    fault: Literal["exception", "before", "after", "none"],
) -> None:
    gate: Final = _CountingGate()
    rig: Final = _Rig(monkeypatch, 1000.0, count=gate.count)
    logging_obj: Final = await rig.native()
    prepared: Final = (
        None if retry_owns_reservation else BaselineCacheContext(rig.estimator, await rig.reserve("replacement"))
    )
    rig.store.before_write = gate.wait if operation == "cancel" else None
    finalizing: Final = asyncio.create_task(
        cancel_baseline_cache(logging_obj)
        if operation == "cancel"
        else logging_obj._prepare_baseline_cache_estimate(ModelResponse())  # pyright: ignore[reportPrivateUsage]  # production Logging delegate
    )
    await asyncio.wait_for(gate.started.wait(), timeout=5)
    if retry_owns_reservation:
        await invalidate_baseline_cache(logging_obj, "retried_request")
    replacement: Final = logging_obj.baseline_cache_context if retry_owns_reservation else prepared
    assert replacement is not None
    logging_obj.baseline_cache_context = replacement
    sentinel: Final = (
        None
        if operation == "cancel"
        else unknown_estimate("retried_request" if fault == "none" else "replacement_estimate")
    )
    logging_obj.baseline_cache_estimate = sentinel
    if fault != "none":
        rig.store.arm(fault)
    gate.release.set()
    assert await asyncio.wait_for(finalizing, timeout=5) is (False if operation == "cancel" else None)
    assert logging_obj.baseline_cache_context is replacement and logging_obj.baseline_cache_estimate is sentinel
    rig.clock.now = 1001.0
    if retry_owns_reservation:
        assert _is_unknown(await rig.finish(), "pending_request")
    elif operation == "finalize" and fault == "exception":
        assert _is_unknown(
            await rig.estimator.finalize(
                replacement.reservation, wire=_wire(), request_started_at=rig.clock.now, available_at=rig.clock.now
            ),
            "history_unavailable",
        )


@pytest.mark.parametrize("phase", ("setup", "prepare", "reserve", "register_storage"))
async def test_native_generation_and_success_telemetry_survive_predispatch_estimator_fault(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    rig: Final = _Rig(monkeypatch, 1000.0, lookup_fails=phase == "setup")
    rig.clock.failure_at = 1 if phase == "prepare" else 2 if phase == "reserve" else None
    if phase == "register_storage":
        rig.store.arm("before")
    logging_obj: Final = rig.logging()

    def upstream(request: httpx.Request) -> httpx.Response:
        if phase in ("reserve", "register_storage"):
            assert logging_obj.baseline_cache_context is not None and logging_obj.baseline_cache_context.invalidated
        return _upstream(request)

    with _transport(upstream) as route:
        await _call(None, logging_obj)
        payload: Final = await rig.capture.payload()
    cost: Final = payload["response_cost"]
    assert route.call_count == 1 and payload["status"] == "success" and isinstance(cost, float) and cost > 0
    assert _is_unknown(
        logging_obj.baseline_cache_estimate,
        "state_unavailable" if phase == "register_storage" else "estimator_unavailable",
    )
    assert logging_obj.baseline_cache_context is None


async def _wait_for_context(logging_obj: Logging, *, invalidated: bool = False) -> BaselineCacheContext:
    while (
        logging_obj.baseline_cache_context is None or logging_obj.baseline_cache_context.invalidated is not invalidated
    ):
        await asyncio.sleep(0)
    return logging_obj.baseline_cache_context


@pytest.mark.parametrize("mode", ("cancel", "retry", "cancel_retry", "invalidate"))
async def test_late_registration_cannot_cancel_retry_owned_reservation(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    rig: Final = _Rig(monkeypatch, 1000.0)
    logging_obj: Final = await rig.native() if mode == "invalidate" else rig.logging()
    kwargs: Final = _kwargs(logging_obj)
    await rig.store.lock.acquire()
    active: Final = asyncio.create_task(
        invalidate_baseline_cache(logging_obj, "retried_request")
        if mode == "invalidate"
        else rig.hook.async_pre_call_deployment_hook(kwargs, CallTypes.anthropic_messages)
    )
    original: Final = await asyncio.wait_for(
        _wait_for_context(logging_obj, invalidated=mode == "invalidate"), timeout=5
    )
    retry: Final = (
        asyncio.create_task(rig.hook.async_pre_call_deployment_hook(kwargs, CallTypes.anthropic_messages))
        if mode in ("retry", "cancel_retry")
        else None
    )
    retained: Final = (
        await asyncio.wait_for(_wait_for_context(logging_obj, invalidated=True), timeout=5) if retry else original
    )
    assert retained.reservation is original.reservation
    if mode != "retry":
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
    rig.store.lock.release()
    if mode == "retry":
        await asyncio.wait_for(active, timeout=5)
    if retry:
        await asyncio.wait_for(retry, timeout=5)
    if mode == "cancel":
        assert logging_obj.baseline_cache_context is None
        assert _is_unknown(await rig.finish(), "history_unavailable")
        return
    current: Final = logging_obj.baseline_cache_context
    assert current is not None and current.invalidated
    if retry:
        assert current is retained and _is_unknown(logging_obj.baseline_cache_estimate, "retried_request")
    assert _is_unknown(await rig.finish(), "pending_request")


@pytest.mark.parametrize("wire,completed", ((False, False), (False, True), (True, False)))
async def test_native_stream_logging_without_wire_retains_scope_until_terminal_event(
    monkeypatch: pytest.MonkeyPatch, wire: bool, completed: bool
) -> None:
    rig: Final = _Rig(monkeypatch, 1000.0)
    logging_obj: Final = await rig.native(stream=True)
    if not wire:
        logging_obj.model_call_details.pop("httpx_response")  # pyright: ignore[reportUnknownMemberType]  # remove only the observed native wire
    timestamp: Final = datetime.fromtimestamp(rig.clock.now)  # noqa: DTZ006  # native timing uses naive timestamps
    await PassThroughStreamingHandler._route_streaming_logging_to_handler(  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # actual complete/partial stream logging route
        litellm_logging_obj=logging_obj,
        passthrough_success_handler_obj=PassThroughEndpointLogging(),
        url_route="/v1/messages",
        request_body=_JSON_OBJECT.validate_json('{"model":"claude-sonnet-5","stream":true}'),
        endpoint_type=EndpointType.ANTHROPIC,
        start_time=timestamp,
        raw_bytes=_sse(completed),
        end_time=timestamp,
    )
    assert (await rig.capture.payload())["status"] == "success"
    assert _is_unknown(logging_obj.baseline_cache_estimate, "incomplete_response" if wire else "missing_final_wire")
    if completed:
        assert logging_obj.baseline_cache_context is None
    else:
        assert logging_obj.baseline_cache_context is not None and logging_obj.baseline_cache_context.invalidated
        assert _is_unknown(await rig.finish(), "pending_request")


@pytest.mark.parametrize("terminal", (False, True))
async def test_disconnected_stream_cleanup_retires_only_unfinished_provider_request(
    monkeypatch: pytest.MonkeyPatch, terminal: bool
) -> None:
    rig: Final = _Rig(monkeypatch, 1000.0)
    logging_obj: Final = await rig.native(stream=True)
    rig.stamp(logging_obj, complete=terminal)

    async def response() -> AsyncIterator[bytes]:
        yield b"partial response"

    data: Final = _OBJECTS.validate_python(MappingProxyType({"litellm_logging_obj": logging_obj}))
    await ProxyBaseLLMRequestProcessing._finalize_streaming_generator_cleanup(  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # real outer disconnect cleanup
        request=None, request_data=data, response=response(), client_disconnected=True
    )
    assert (logging_obj.baseline_cache_context is None) is (not terminal)
    if not terminal:
        assert _is_unknown(logging_obj.baseline_cache_estimate, "incomplete_response")
        assert _is_unknown(await rig.finish(), "history_unavailable")


@pytest.mark.parametrize("retries", (0, 1))
@pytest.mark.parametrize("cleanup_fails", (False, True))
async def test_terminal_proxy_failure_retires_reservation_after_router_exhaustion(
    monkeypatch: pytest.MonkeyPatch, retries: int, cleanup_fails: bool
) -> None:
    rig: Final = _Rig(monkeypatch, round(time.time(), 6), retries)
    shared: Final = rig.logging()
    with (
        _transport(lambda request: _error(request, 429, "terminal provider failure")) as route,
        pytest.raises(litellm.RateLimitError, match="terminal provider failure") as caught,
    ):
        await _call(rig.router, shared)
    assert route.call_count == retries + 1 and shared.baseline_cache_context is not None
    proxy: Final = ProxyLogging(UserApiKeyCache())
    proxy.alert_types = []  # mutable-ok: isolate optional alert sinks
    data: Final = _OBJECTS.validate_python(
        MappingProxyType({"model": "test-router", "litellm_call_id": "first", "litellm_logging_obj": shared})
    )
    rig.clock.failures_remaining = 2 if cleanup_fails else 0
    transformed: Final = await proxy.post_call_failure_hook(  # pyright: ignore[reportUnknownMemberType]  # actual proxy terminal owner
        request_data=data,
        original_exception=caught.value,
        user_api_key_dict=UserAPIKeyAuth(request_route="/v1/messages"),
    )
    assert transformed is rig.capture.replacement_error
    assert await asyncio.wait_for(rig.capture.terminal_errors.get(), timeout=5) is caught.value
    assert "litellm_logging_obj" not in data and shared.baseline_cache_context is None
    await rig.recover()
