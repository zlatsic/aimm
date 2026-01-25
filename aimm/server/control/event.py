import base64
import contextlib
import logging

from hat import aio
import hat.event.common

from aimm.server import common
from aimm import plugins


mlog = logging.getLogger(__name__)


def create_subscription(conf):
    return [tuple([*p, "call", "*"]) for p in conf["event_prefixes"].values()]


async def create(conf, engine, event_client):
    common.json_schema_validator.validate(
        "aimm://server/control/event.yaml#", conf
    )
    if event_client is None:
        raise ValueError(
            "attempting to create event control without hat compatibility"
        )
    return EventControl(conf, engine, event_client)


def _log_exception(error_log):
    def inner(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ValueError as e:
                mlog.warning(error_log, exc_info=e)
            except Exception as e:
                mlog.error(error_log, exc_info=e)
        return wrapper
    return inner


class EventControl(common.Control):
    def __init__(self, conf: dict, engine: common.Engine, event_client):
        self._client = event_client
        self._engine = engine
        self._async_group = aio.Group()
        self._event_prefixes = conf["event_prefixes"]
        self._executor = aio.create_executor()
        self._notified_state = {}
        self._in_progress = {}

    @property
    def async_group(self) -> aio.Group:
        """Async group"""
        return self._async_group

    async def process_events(self, events):
        for event in events:
            self.async_group.spawn(self._process_action, event)

    async def _process_action(self, event):
        if self._prefix_match("create_instance", event):
            await self._create_instance(event)
        if self._prefix_match("add_instance", event):
            await self._add_instance(event)
        if self._prefix_match("update_instance", event):
            await self._update_instance(event)
        if self._prefix_match("fit", event):
            await self._fit(event)
        if self._prefix_match("predict", event):
            await self._predict(event)
        if self._prefix_match("cancel", event):
            self._cancel(event)

    def _prefix_match(self, action_prefix, event):
        if action_prefix not in self._event_prefixes:
            return False
        # For actions that need instance_id (fit, predict, update_instance), 
        # the pattern is [prefix, instance_id, "call", "*"]
        if action_prefix in ["fit", "predict", "update_instance"]:
            prefix = self._event_prefixes[action_prefix]
            if len(event.type) >= len(prefix) + 3:
                return (list(event.type[:len(prefix)]) == prefix and 
                        list(event.type[-2:]) == ["call", "*"])
        # For other actions, the pattern is [prefix, "call", "*"]
        return hat.event.common.matches_query_type(
            event.type, self._event_prefixes[action_prefix] + ["call", "*"]
        )

    @_log_exception("instance creation failed")
    async def _create_instance(self, event):
        data = event.payload.data

        model_type = data["model_type"]
        args = [await _process_arg(arg) for arg in data["args"]]
        kwargs = {k: await _process_arg(v) for k, v in data["kwargs"].items()}

        action = self._engine.create_instance(model_type, *args, **kwargs)
        with self._action_context(event, action):
            await self._register_result(event, await action.wait_result())

    @_log_exception("add instance failed")
    async def _add_instance(self, event):
        data = event.payload.data
        instance = await self._instance_from_json(
            data["instance"], data["model_type"]
        )
        instance_id = await self._engine.add_instance(
            data["model_type"], instance
        )
        await self._register_result(event, instance_id)

    @_log_exception("update instance failed")
    async def _update_instance(self, event):
        event_prefix = self._event_prefixes.get("update_instance")
        instance_id = int(event.type[len(event_prefix)])

        data = event.payload.data
        model_type = data["model_type"]
        instance = await self._instance_from_json(data["instance"], model_type)

        model = common.Model(
            model_type=data["model_type"],
            instance_id=instance_id,
        )
        await self._engine.update_instance(model, instance)

    @_log_exception("fitting failed")
    async def _fit(self, event):
        event_prefix = self._event_prefixes["fit"]
        instance_id = int(event.type[len(event_prefix)])

        data = event.payload.data
        args = [await _process_arg(a) for a in data["args"]]
        kwargs = {k: await _process_arg(v) for k, v in data["kwargs"].items()}

        action = self._engine.fit(instance_id, *args, **kwargs)
        with self._action_context(event, action):
            await action.wait_result()

    @_log_exception("prediction failed")
    async def _predict(self, event):
        event_prefix = self._event_prefixes["predict"]
        instance_id = int(event.type[len(event_prefix)])

        data = event.payload.data
        args = [await _process_arg(a) for a in data["args"]]
        kwargs = {k: await _process_arg(v) for k, v in data["kwargs"].items()}

        action = self._engine.predict(instance_id, *args, **kwargs)
        with self._action_context(event, action):
            await self._register_result(event, await action.wait_result())

    def _cancel(self, event):
        request_event_id = event.payload.data
        if request_event_id in self._in_progress:
            self._in_progress[request_event_id].close()

    @contextlib.contextmanager
    def _action_context(self, event, action):
        data = event.payload.data
        with action.subscribe_to_state_change(
            lambda state: self.async_group.spawn(
                self._register_action_state, event, state
            )
        ):
            self._in_progress[data["request_id"]] = action
            try:
                yield
            finally:
                del self._in_progress[data["request_id"]]

    async def _register_action_state(
        self, call_event, state: common.ActionState
    ):
        state_type = list(call_event.type)
        state_type[-2] = "state"
        return await self._client.register(
            [
                _register_event(
                    tuple(state_type),
                    {
                        "request_id": call_event.payload.data["request_id"],
                        "state": _action_to_json(state),
                    },
                )
            ]
        )

    async def _register_result(self, call_event, result):
        result_type = list(call_event.type)
        result_type[-2] = "result"
        return await self._client.register(
            [_register_event(tuple(result_type), result)]
        )

    async def _instance_from_json(self, instance_b64, model_type):
        return await self._executor(
            plugins.exec_deserialize,
            model_type,
            base64.b64decode(instance_b64),
        )


async def _process_arg(arg):
    if not (isinstance(arg, dict) and arg.get("type") == "data_access"):
        return arg
    return plugins.DataAccessArg(
        name=arg["name"], args=arg["args"], kwargs=arg["kwargs"]
    )


def _action_to_json(action: common.ActionState):
    action_dict = action._asdict()
    action_dict["run"] = _execution_to_json(action.run)
    return action_dict


def _execution_to_json(execution_state: plugins.ExecutionState):
    if execution_state is None:
        return None
    run_dict = execution_state._asdict()
    run_dict["data_access"] = {
        k: _execution_to_json(v)
        for k, v in execution_state.data_access.items()
    }
    run_dict["action"] = _execution_to_json(execution_state.action)
    return run_dict


def _register_event(event_type, payload, source_timestamp=None):
    return hat.event.common.RegisterEvent(
        type=tuple(event_type),
        source_timestamp=source_timestamp,
        payload=hat.event.common.EventPayloadJson(payload),
    )
