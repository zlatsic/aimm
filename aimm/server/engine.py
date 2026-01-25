import asyncio
import logging
from typing import Callable, Coroutine, List

from hat import aio
from hat import util

from aimm import plugins
from aimm.server import common
from aimm.server import mprocess
from aimm.server.common import ActionState, Model

mlog = logging.getLogger(__name__)


class Engine(common.Engine):
    """Engine implementation
    Args:
        conf: configuration that follows schema with id
            ``aimm://server/schema.yaml#/definitions/engine``
        backend: backend
    """

    def __init__(self, conf: dict, backend: common.Backend):
        self._group = aio.Group()
        self._backend = backend

        self._pool = mprocess.ProcessManager(
            conf["max_children"],
            self._group.create_subgroup(),
            conf["check_children_period"],
            conf["sigterm_timeout"],
        )

    @property
    def async_group(self):
        return self._group

    async def scan_models(self) -> List[Model]:
        return await self._backend.scan_models()

    def create_instance(self, model_type, *args, **kwargs):
        state_mgr = _StateManager({
            "call": "create_instance",
            "model_type": model_type,
            "args": [str(a) for a in args],
            "kwargs": {k: str(v) for k, v in kwargs.items()},
        })

        @_wrap_action(state_mgr)
        async def action():
            handler = self._pool.create_handler(state_mgr.set_run)
            state_mgr.set_status(common.ActionStatus.RUNNING)
            try:
                instance = await handler.run(
                    plugins.exec_instantiate,
                    model_type,
                    handler.proc_notify_state_change,
                    *args,
                    **kwargs
                )

                state_mgr.set_status(common.ActionStatus.STORING)
                return await self._backend.create_model(model_type, instance)
            finally:
                await handler.async_close()

        return _Action(self._group.create_subgroup(), state_mgr, action)

    async def add_instance(self, model_type, instance):
        return await self._backend.create_model(model_type, instance)

    async def update_instance(self, model, instance):
        await self._backend.update_instance(model, instance)

    def fit(self, instance_id, *args, **kwargs):
        state_mgr = _StateManager({
            "call": "fit",
            "model": instance_id,
            "args": [str(a) for a in args],
            "kwargs": {k: str(v) for k, v in kwargs.items()},
        })

        @_wrap_action(state_mgr)
        async def action():
            handler = self._pool.create_handler(state_mgr.set_run)
            state_mgr.set_status(common.ActionStatus.RUNNING)
            try:
                model_type, instance = await self._backend.get_instance(
                    instance_id,
                )
                instance = await handler.run(
                    plugins.exec_fit,
                    model_type,
                    instance,
                    handler.proc_notify_state_change,
                    *args,
                    **kwargs
                )

                state_mgr.set_status(common.ActionStatus.STORING)
                await self._backend.update_instance(
                    common.Model(
                        model_type=model_type, instance_id=instance_id
                    ),
                    instance,
                )
            finally:
                await handler.async_close()

        return _Action(self._group.create_subgroup(), state_mgr, action)

    def predict(self, instance_id, *args, **kwargs):
        state_mgr = _StateManager({
            "call": "predict",
            "model": instance_id,
            "args": [str(a) for a in args],
            "kwargs": {k: str(v) for k, v in kwargs.items()},
        })

        @_wrap_action(state_mgr)
        async def action():
            handler = self._pool.create_handler(state_mgr.set_run)
            try:
                model_type, instance = await self._backend.get_instance(
                    instance_id,
                )
                state_mgr.set_status(common.ActionStatus.RUNNING)
                instance, prediction = await handler.run(
                    plugins.exec_predict,
                    model_type,
                    instance,
                    handler.proc_notify_state_change,
                    *args,
                    **kwargs
                )

                state_mgr.set_status(common.ActionStatus.STORING)
                await self._backend.update_instance(
                    common.Model(
                        model_type=model_type, instance_id=instance_id
                    ),
                    instance,
                )
                return prediction
            finally:
                await handler.async_close()

        return _Action(self._group.create_subgroup(), state_mgr, action)


class _Action(common.Action):
    def __init__(
        self, async_group, state: "_StateManager", call: Callable
    ):
        self._group = async_group
        self._state = state

        self._task = self._group.spawn(self._run, call)

    @property
    def async_group(self):
        return self._group

    def subscribe_to_state_change(
        self,
        state_cb: Callable[[ActionState], None],
    ) -> util.RegisterCallbackHandle:
        return self._state.subscribe_to_state_change(state_cb)

    async def wait_result(self):
        result = await self._task
        await self.async_close()
        return result

    async def _run(self, call: Callable):
        return await aio.call(call)


class _StateManager:
    def __init__(self, meta: dict):
        self._meta = meta
        self._state = None
        self._callback_registry = util.CallbackRegistry()

    def initialize(self):
        self._state = common.ActionState(
            meta=self._meta,
            status=common.ActionStatus.INIT,
            run=None,
        )
        self._callback_registry.notify(self._state)

    def set_status(self, status: common.ActionStatus):
        if not isinstance(self._state, common.ActionState):
            raise Exception("state not initialized")
        self._state = self._state._replace(status=status)
        self._callback_registry.notify(self._state)

    def set_run(self, run: plugins.ExecutionState):
        if not isinstance(self._state, common.ActionState):
            raise Exception("state not initialized")
        self._state = self._state._replace(run=run)
        self._callback_registry.notify(self._state)

    def subscribe_to_state_change(
        self,
        cb: Callable[[common.ActionState], None],
    ):
        return self._callback_registry.register(cb)


def _wrap_action(state_manager: _StateManager):
    def inner(func: Callable[[], Coroutine]):
        async def action():
            state_manager.initialize()
            try:
                result = await func()
                state_manager.set_status(common.ActionStatus.COMPLETE)
                return result
            except asyncio.CancelledError:
                state_manager.set_status(common.ActionStatus.CANCELLED)
            except Exception:
                state_manager.set_status(common.ActionStatus.ERROR)
                raise
        return action
    return inner
