from functools import partial
import asyncio
import itertools
import logging
import typing

from hat import aio
from hat import util

from aimm import plugins
from aimm.server import common
from aimm.server import mprocess


mlog = logging.getLogger(__name__)


async def create(conf: typing.Dict, backend: common.Backend) -> common.Engine:
    """Create engine

    Args:
        conf: configuration that follows schema with id
            ``aimm://server/schema.yaml#/definitions/engine``
        backend: backend

    Returns:
        engine
    """
    engine = _Engine(conf, backend)
    await engine.start()
    return engine


class _Engine(common.Engine):
    """Engine implementation, use :func:`create` to instantiate"""

    def __init__(self, conf, backend):
        self._group = aio.Group()
        self._backend = backend
        self._conf = conf
        self._state = {"actions": {}, "models": {}}
        self._locks = {
            instance_id: asyncio.Lock()
            for instance_id in self._state["models"]
        }

        self._action_id_gen = itertools.count(1)

        self._pool = mprocess.ProcessManager(
            conf["max_children"],
            self._group.create_subgroup(),
            conf["check_children_period"],
            conf["sigterm_timeout"],
        )
        self._callback_registry = util.CallbackRegistry()

    @property
    def async_group(self):
        return self._group

    @property
    def state(self):
        return self._state

    async def start(self):
        models = await self._backend.get_models()
        self._state = {
            "actions": {},
            "models": {model.instance_id: model for model in models},
        }

    def subscribe_to_state_change(self, cb):
        return self._callback_registry.register(cb)

    def create_instance(self, model_type, *args, **kwargs):
        action_id = next(self._action_id_gen)
        state_cb = partial(self._update_action, action_id)
        return create_action(
            self._group.create_subgroup(),
            self._act_create_instance,
            model_type,
            args,
            kwargs,
            state_cb,
        )

    async def add_instance(self, model_type, instance):
        model = await self._backend.create_model(model_type, instance)
        self._set_model(model)
        return model

    async def update_instance(self, model: common.Model):
        """Update existing instance in the state"""
        self._set_model(model)
        await self._backend.update_model(model)

    def fit(self, instance_id, *args, **kwargs):
        action_id = next(self._action_id_gen)
        state_cb = partial(self._update_action, action_id)
        return create_action(
            self._group.create_subgroup(),
            self._act_fit,
            instance_id,
            args,
            kwargs,
            state_cb,
        )

    def predict(self, instance_id, *args, **kwargs):
        action_id = next(self._action_id_gen)
        state_cb = partial(self._update_action, action_id)
        return create_action(
            self._group.create_subgroup(),
            self._act_predict,
            instance_id,
            args,
            kwargs,
            state_cb,
        )

    def _update_action(self, action_id, action_state):
        actions = dict(self.state["actions"])
        actions.update({action_id: action_state})
        self._update_state(dict(self.state, actions=actions))

    def _set_model(self, model):
        if model.instance_id not in self._locks:
            self._locks[model.instance_id] = asyncio.Lock()
        models = dict(self.state["models"])
        models.update({model.instance_id: model})
        self._update_state(dict(self.state, models=models))

    def _update_state(self, new_state):
        self._state = new_state
        self._callback_registry.notify()

    async def _act_create_instance(self, model_type, args, kwargs, state_cb):
        with _StateManager(
            {
                "call": "create_instance",
                "model_type": model_type,
                "args": [str(a) for a in args],
                "kwargs": {k: str(v) for k, v in kwargs.items()},
            },
            state_cb,
        ) as state:

            handler = self._pool.create_handler(state.set_run)
            instance = await handler.run(
                plugins.exec_instantiate,
                model_type,
                handler.proc_notify_state_change,
                *args,
                **kwargs
            )
            state.set_status("storing")

            model = await self._backend.create_model(model_type, instance)
            self._set_model(model)
        return model

    async def _act_fit(self, instance_id, args, kwargs, state_cb):
        with _StateManager(
            {
                "call": "fit",
                "model": instance_id,
                "args": [str(a) for a in args],
                "kwargs": {k: str(v) for k, v in kwargs.items()},
            },
            state_cb,
        ) as state:

            handler = self._pool.create_handler(state.set_run)
            model = self.state["models"][instance_id]
            async with self._locks[instance_id]:
                instance = await handler.run(
                    plugins.exec_fit,
                    model.model_type,
                    model.instance,
                    handler.proc_notify_state_change,
                    *args,
                    **kwargs
                )

            state.set_status("storing")
            return await self._update_model(instance, model)

    async def _act_predict(self, instance_id, args, kwargs, state_cb):
        with _StateManager(
            {
                "meta": {
                    "call": "predict",
                    "model": instance_id,
                    "args": [str(a) for a in args],
                    "kwargs": {k: str(v) for k, v in kwargs.items()},
                },
                "status": "running",
                "run": None,
            },
            state_cb,
        ) as state:
            handler = self._pool.create_handler(state.set_run)
            async with self._locks[instance_id]:
                model = self.state["models"][instance_id]
                instance, prediction = await handler.run(
                    plugins.exec_predict,
                    model.model_type,
                    model.instance,
                    handler.proc_notify_state_change,
                    *args,
                    **kwargs
                )

            state.set_status("storing")
            await self._update_model(instance, model)
            return prediction

    async def _update_model(self, instance, model):
        new_model = common.Model(
            instance=instance,
            model_type=model.model_type,
            instance_id=model.instance_id,
        )
        await self._backend.update_model(new_model)

        self._set_model(new_model)
        return new_model


def create_action(
    async_group: aio.Group, fn: typing.Callable, *args, **kwargs
) -> common.Action:
    return _Action(async_group, fn, *args, **kwargs)


class _Action(common.Action):
    def __init__(self, async_group, fn, *args, **kwargs):
        self._group = async_group
        self._task = self._group.spawn(fn, *args, **kwargs)

    @property
    def async_group(self):
        return self._group

    async def wait_result(self):
        return await self._task


class _StateManager:
    def __init__(self, meta, state_cb):
        self._state_cb = state_cb
        self._state = {"meta": meta}

    def set_status(self, status):
        self._set("status", status)

    def set_run(self, run):
        self._set("run", run)

    def _set(self, key: str, value):
        self._state = {**self._state, key: value}
        self._state_cb(self._state)

    def __enter__(self) -> "_StateManager":
        self.set_status("running")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.set_status("error" if exc_type else "complete")
