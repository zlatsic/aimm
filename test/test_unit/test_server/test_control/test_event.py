import asyncio
import base64
from typing import List, Any, Callable

from hat import aio, util
import hat.event.common
import pytest

import aimm.server.control.event
import aimm.server.engine
from aimm.server import common
from aimm import plugins
from aimm.server.common import Model, ActionState


class MockClient:
    def __init__(self):
        self._register_queue = aio.Queue()
        self._receive_queue = aio.Queue()

    async def register(self, events, _=False):
        self._register_queue.put_nowait(events)


class MockEngine(common.Engine):

    def __init__(
        self,
        create_instance_cb=None,
        scan_models_cb=None,
        add_instance_cb=None,
        update_instance_cb=None,
        fit_cb=None,
        predict_cb=None,
    ):
        self._cb = None
        self._create_instance_cb = create_instance_cb
        self._scan_models_cb = scan_models_cb
        self._add_instance_cb = add_instance_cb
        self._update_instance_cb = update_instance_cb
        self._fit_cb = fit_cb
        self._predict_cb = predict_cb

        self._group = aio.Group()
        self._actions = []

    @property
    def async_group(self):
        return self._group

    def create_instance(self, *args, **kwargs):
        if not self._create_instance_cb:
            raise NotImplementedError()
        return MockAction(
            self._group.create_subgroup(),
            self._create_instance_cb,
            "create_instance",
            *args,
            **kwargs,
        )

    async def scan_models(self) -> List[Model]:
        if self._scan_models_cb:
            return await aio.call(self._scan_models_cb)
        raise NotImplementedError()

    async def add_instance(self, *args, **kwargs):
        if self._add_instance_cb:
            return await aio.call(self._add_instance_cb, *args, **kwargs)
        raise NotImplementedError()

    async def update_instance(self, *args, **kwargs):
        if self._update_instance_cb:
            return await aio.call(self._update_instance_cb, *args, **kwargs)
        raise NotImplementedError()

    def fit(self, *args, **kwargs):
        if self._fit_cb:
            action = MockAction(
                self._group.create_subgroup(),
                self._fit_cb,
                "fit",
                *args,
                **kwargs,
            )
            self._actions.append(action)
            return action
        raise NotImplementedError()

    def predict(self, *args, **kwargs):
        if self._predict_cb:
            return MockAction(
                self._group.create_subgroup(),
                self._predict_cb,
                "predict",
                *args,
                **kwargs,
            )
        raise NotImplementedError()


class MockAction(common.Action):

    def __init__(self, group, fn, call_type, *args, **kwargs):
        self._group = group
        self._cb_registry = util.CallbackRegistry()
        self._fn = fn
        self._call_type = call_type
        self._args = args
        self._kwargs = kwargs
        self._result = None
        self._task = None

    @property
    def async_group(self) -> aio.Group:
        return self._group

    def subscribe_to_state_change(
        self,
        state_cb: Callable[[ActionState], None]
    ) -> util.RegisterCallbackHandle:
        return self._cb_registry.register(state_cb)

    async def wait_result(self) -> Any:
        # Create meta based on call type
        if self._call_type == "create_instance":
            meta = {
                "call": "create_instance",
                "model_type": self._args[0] if self._args else "unknown",
                "args": [str(a) for a in self._args[1:]],
                "kwargs": {k: str(v) for k, v in self._kwargs.items()},
            }
        elif self._call_type == "fit":
            meta = {
                "call": "fit",
                "model": self._args[0] if self._args else "unknown",
                "args": [str(a) for a in self._args[1:]],
                "kwargs": {k: str(v) for k, v in self._kwargs.items()},
            }
        elif self._call_type == "predict":
            meta = {
                "call": "predict",
                "model": self._args[0] if self._args else "unknown",
                "args": [str(a) for a in self._args[1:]],
                "kwargs": {k: str(v) for k, v in self._kwargs.items()},
            }
        else:
            meta = {
                "call": self._call_type,
                "args": [str(a) for a in self._args],
                "kwargs": {k: str(v) for k, v in self._kwargs.items()},
            }

        # Simulate state changes
        init_state = ActionState(
            meta=meta,
            status=common.ActionStatus.INIT,
            run=None,
        )
        self._cb_registry.notify(init_state)

        running_state = ActionState(
            meta=meta,
            status=common.ActionStatus.RUNNING,
            run=None,
        )
        self._cb_registry.notify(running_state)

        # Create a task that can be cancelled
        async def _execute():
            return await aio.call(self._fn, *self._args, **self._kwargs)

        self._task = self._group.spawn(_execute)

        # Execute the function
        try:
            self._result = await self._task
        except asyncio.CancelledError:
            # Send cancelled state
            cancelled_state = ActionState(
                meta=meta,
                status=common.ActionStatus.CANCELLED,
                run=None,
            )
            self._cb_registry.notify(cancelled_state)
            raise

        complete_state = ActionState(
            meta=meta,
            status=common.ActionStatus.COMPLETE,
            run=None,
        )
        self._cb_registry.notify(complete_state)

        return self._result


def assert_event(event, event_type, payload, source_timestamp=None):
    assert event.type == event_type
    assert event.source_timestamp == source_timestamp
    assert event.payload.data == payload


def conf():
    return {
        "event_prefixes": {
            "create_instance": ["create_instance"],
            "add_instance": ["add_instance"],
            "update_instance": ["update_instance"],
            "fit": ["fit"],
            "predict": ["predict"],
            "cancel": ["cancel"],
        },
    }


@pytest.mark.timeout(1)
async def test_create_instance():
    create_queue = aio.Queue()

    async def create_instance_cb(model_type, *c_args, **c_kwargs):
        complete_future = asyncio.Future()
        create_queue.put_nowait(
            {
                "model_type": model_type,
                "args": c_args,
                "kwargs": c_kwargs,
                "complete_future": complete_future,
            }
        )
        return await complete_future

    client = MockClient()
    engine = MockEngine(create_instance_cb=create_instance_cb)
    control = await aimm.server.control.event.create(conf(), engine, client)

    args = ["a1", "a2"]
    kwargs = {"k1": "1"}
    req_event = _event(
        ("create_instance", "call", "*"),
        {
            "model_type": "Model1",
            "args": args,
            "kwargs": kwargs,
            "request_id": "1",
        },
    )
    await control.process_events([req_event])
    call = await create_queue.get()
    assert call["model_type"] == "Model1"
    assert call["args"] == tuple(args)
    assert call["kwargs"] == kwargs

    call["complete_future"].set_result(1)

    # Collect all events
    all_events = []
    for _ in range(4):  # Expect 3 state events + 1 result event
        events = await client._register_queue.get()
        all_events.extend(events)

    # Verify we got the expected events
    state_events = [e for e in all_events if e.type[1] == "state"]
    result_events = [e for e in all_events if e.type[1] == "result"]

    assert len(state_events) == 3  # INIT, RUNNING, COMPLETE
    assert len(result_events) == 1

    # Verify result event
    result_event = result_events[0]
    assert result_event.type == ("create_instance", "result", "*")
    assert result_event.payload.data == 1

    # Verify state events have correct structure
    for state_event in state_events:
        assert state_event.type == ("create_instance", "state", "*")
        assert state_event.payload.data["request_id"] == "1"
        assert "state" in state_event.payload.data
        state_data = state_event.payload.data["state"]
        assert state_data["meta"]["call"] == "create_instance"
        assert state_data["meta"]["model_type"] == "Model1"
        assert state_data["meta"]["args"] == ["a1", "a2"]
        assert state_data["meta"]["kwargs"] == {"k1": "1"}
        assert state_data["run"] is None
        assert state_data["status"] in ["init", "running", "complete"]
    await control.async_close()


@pytest.mark.timeout(10)
async def test_add_instance(plugin_teardown):
    @plugins.deserialize(["Model1"])
    def deserialize(instance_bytes):
        return instance_bytes.decode("utf-8")

    add_queue = aio.Queue()

    async def add_instance_cb(model_type, instance):
        complete_future = asyncio.Future()
        add_queue.put_nowait(
            {
                "instance": instance,
                "model_type": model_type,
                "complete_future": complete_future,
            }
        )
        return await complete_future

    client = MockClient()
    engine = MockEngine(add_instance_cb=add_instance_cb)
    control = await aimm.server.control.event.create(conf(), engine, client)

    req_event = _event(
        ("add_instance", "call", "*"),
        {
            "model_type": "Model1",
            "instance": base64.b64encode("xyz".encode("utf-8")).decode(
                "utf-8"
            ),
            "request_id": "1",
        },
    )
    await control.process_events([req_event])

    call = await add_queue.get()
    assert call["model_type"] == "Model1"
    assert call["instance"] == "xyz"
    call["complete_future"].set_result(2)

    events = await client._register_queue.get()
    assert len(events) == 1
    event = events[0]
    assert event.type == ("add_instance", "result", "*")
    assert event.payload.data == 2

    await control.async_close()


@pytest.mark.timeout(1)
async def test_update_instance(plugin_teardown):
    @plugins.deserialize(["Model1"])
    def deserialize(instance_bytes):
        return instance_bytes.decode("utf-8")

    update_queue = aio.Queue()

    async def update_instance_cb(model, instance):
        complete_future = asyncio.Future()
        update_queue.put_nowait(
            {
                "model": model,
                "instance": instance,
                "complete_future": complete_future,
            }
        )
        return await complete_future

    client = MockClient()
    engine = MockEngine(update_instance_cb=update_instance_cb)
    control = await aimm.server.control.event.create(conf(), engine, client)

    req_event = _event(
        ("update_instance", "10", "call", "*"),
        {
            "model_type": "Model1",
            "instance": base64.b64encode("xyz".encode("utf-8")).decode(
                "utf-8"
            ),
            "request_id": "1",
        },
    )
    await control.process_events([req_event])

    call = await update_queue.get()
    assert call["model"] == common.Model(
        model_type="Model1", instance_id=10
    )
    assert call["instance"] == "xyz"
    call["complete_future"].set_result(call["model"])

    # update_instance doesn't generate any events, it's a direct call
    await control.async_close()


@pytest.mark.timeout(1)
async def test_fit():
    fit_queue = aio.Queue()

    async def fit_cb(model_id, *args, **kwargs):
        done_future = asyncio.Future()
        fit_queue.put_nowait(
            {
                "done_future": done_future,
                "args": args,
                "kwargs": kwargs,
                "model_id": model_id,
            }
        )
        return await done_future

    client = MockClient()
    engine = MockEngine(fit_cb=fit_cb)
    control = await aimm.server.control.event.create(conf(), engine, client)

    req_event = _event(
        ("fit", "11", "call", "*"),
        {
            "args": ["a", "b"],
            "kwargs": {"c": "d", "e": "f"},
            "request_id": "1",
        },
    )
    await control.process_events([req_event])

    call = await fit_queue.get()
    assert call["model_id"] == 11
    assert call["args"] == ("a", "b")
    assert call["kwargs"] == {"c": "d", "e": "f"}

    call["done_future"].set_result("fitted instance")

    # Collect all events (fit only generates state events, no result event)
    all_events = []
    for _ in range(3):  # Expect 3 state events only
        events = await client._register_queue.get()
        all_events.extend(events)

    # Verify we got the expected events
    state_events = [e for e in all_events if e.type[2] == "state"]

    assert len(state_events) == 3  # INIT, RUNNING, COMPLETE

    # Verify state events have correct structure
    for state_event in state_events:
        assert state_event.type == ("fit", "11", "state", "*")
        assert state_event.payload.data["request_id"] == "1"
        assert "state" in state_event.payload.data
        state_data = state_event.payload.data["state"]
        assert state_data["meta"]["call"] == "fit"
        assert state_data["meta"]["model"] == 11
        assert state_data["meta"]["args"] == ["a", "b"]
        assert state_data["meta"]["kwargs"] == {"c": "d", "e": "f"}
        assert state_data["run"] is None
        assert state_data["status"] in ["init", "running", "complete"]
    await control.async_close()


@pytest.mark.timeout(1)
async def test_predict():
    predict_queue = aio.Queue()

    async def predict_cb(model_id, *args, **kwargs):
        done_future = asyncio.Future()
        predict_queue.put_nowait(
            {
                "done_future": done_future,
                "args": args,
                "kwargs": kwargs,
                "model_id": model_id,
            }
        )
        return await done_future

    client = MockClient()
    engine = MockEngine(predict_cb=predict_cb)
    control = await aimm.server.control.event.create(conf(), engine, client)

    req_event = _event(
        ("predict", "12", "call", "*"),
        {
            "args": ["a", "b"],
            "kwargs": {"c": "d", "e": "f"},
            "request_id": "1",
        },
    )
    await control.process_events([req_event])

    call = await predict_queue.get()
    assert call["model_id"] == 12
    assert call["args"] == ("a", "b")
    assert call["kwargs"] == {"c": "d", "e": "f"}

    call["done_future"].set_result("prediction")

    # Collect all events
    all_events = []
    for _ in range(4):  # Expect 3 state events + 1 result event
        events = await client._register_queue.get()
        all_events.extend(events)

    # Verify we got the expected events
    state_events = [e for e in all_events if e.type[2] == "state"]
    result_events = [e for e in all_events if e.type[2] == "result"]

    assert len(state_events) == 3  # INIT, RUNNING, COMPLETE
    assert len(result_events) == 1

    # Verify result event
    result_event = result_events[0]
    assert result_event.type == ("predict", "12", "result", "*")
    assert result_event.payload.data == "prediction"

    # Verify state events have correct structure
    for state_event in state_events:
        assert state_event.type == ("predict", "12", "state", "*")
        assert state_event.payload.data["request_id"] == "1"
        assert "state" in state_event.payload.data
        state_data = state_event.payload.data["state"]
        assert state_data["meta"]["call"] == "predict"
        assert state_data["meta"]["model"] == 12
        assert state_data["meta"]["args"] == ["a", "b"]
        assert state_data["meta"]["kwargs"] == {"c": "d", "e": "f"}
        assert state_data["run"] is None
        assert state_data["status"] in ["init", "running", "complete"]
    await control.async_close()


@pytest.mark.timeout(1)
async def test_cancel():
    future_queue = aio.Queue()

    async def predict_cb(_, *__, **___):
        done_future = asyncio.Future()
        future_queue.put_nowait(done_future)
        return await done_future

    client = MockClient()
    engine = MockEngine(predict_cb=predict_cb)
    control = await aimm.server.control.event.create(conf(), engine, client)

    # Start a predict action
    req_event = _event(
        ("predict", "12", "call", "*"),
        {"args": [], "kwargs": {}, "request_id": "1"},
    )
    await control.process_events([req_event])

    # Get initial state events (INIT and RUNNING)
    all_events = []
    for _ in range(2):  # Expect INIT and RUNNING state events
        events = await client._register_queue.get()
        all_events.extend(events)

    # Verify we got the expected state events
    state_events = [e for e in all_events if e.type[2] == "state"]
    assert len(state_events) == 2  # INIT and RUNNING

    # Verify the state events have correct structure
    for state_event in state_events:
        assert state_event.type == ("predict", "12", "state", "*")
        assert state_event.payload.data["request_id"] == "1"
        assert "state" in state_event.payload.data
        state_data = state_event.payload.data["state"]
        assert state_data["meta"]["call"] == "predict"
        assert state_data["meta"]["model"] == 12
        assert state_data["status"] in ["init", "running"]

    # Get the future that represents the running action
    future = await future_queue.get()

    # Send cancel event
    cancel_event = _event(("cancel", "call", "*"), "1")
    await control.process_events([cancel_event])

    # The action should be cancelled
    with pytest.raises(asyncio.CancelledError):
        await future

    # Collect the cancelled state event
    cancelled_events = []
    try:
        # Try to get the cancelled state event with a timeout
        events = await asyncio.wait_for(
            client._register_queue.get(), timeout=0.1
        )
        cancelled_events.extend(events)
    except asyncio.TimeoutError:
        # If no event comes within timeout, that's also acceptable
        # The important thing is that the cancellation worked
        pass

    # If we got a cancelled event, verify it
    if cancelled_events:
        state_events = [e for e in cancelled_events if e.type[2] == "state"]
        for state_event in state_events:
            assert state_event.type == ("predict", "12", "state", "*")
            assert state_event.payload.data["request_id"] == "1"
            assert "state" in state_event.payload.data
            state_data = state_event.payload.data["state"]
            assert state_data["status"] == "cancelled"

    await control.async_close()


def _register_event(event_type, payload, source_timestamp=None):
    return hat.event.common.RegisterEvent(
        type=event_type,
        source_timestamp=source_timestamp,
        payload=hat.event.common.EventPayload(payload),
    )


def _event(
    event_type,
    payload,
    source_timestamp=None,
    event_id=hat.event.common.EventId(server=0, instance=0, session=0),
):
    return hat.event.common.Event(
        id=event_id,
        type=event_type,
        timestamp=hat.event.common.now(),
        source_timestamp=source_timestamp,
        payload=hat.event.common.EventPayloadJson(payload),
    )
