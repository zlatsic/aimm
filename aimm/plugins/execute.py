"""Depending on the plugin implementation, potentially long-lasting methods
performing:
* dataset downloads
* model optimizations

Meant to be executed in parallel for in-production systems.
"""

from typing import Any, ByteString, Union

from aimm.plugins import common
from aimm.plugins import decorators


def exec_data_access(
    name: str,
    state_cb: common.StateCallback = lambda state: None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Uses a loaded plugin to access data"""
    with _StateManager(state_cb) as state:
        plugin = decorators.get_data_access(name)

        args, kwargs = _preprocess_args(args, kwargs, plugin, state)

        state.set_status(common.ExecutionStatus.RUNNING)
        return plugin.function(*args, **kwargs)


def exec_instantiate(
    model_type: str,
    state_cb: common.StateCallback = lambda state: None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Uses a loaded plugin to create a model instance. Will handle any data
    access arguments and load the data."""
    with _StateManager(state_cb) as state:
        plugin = decorators.get_instantiate(model_type)

        args, kwargs = _preprocess_args(args, kwargs, plugin, state)

        state.set_status(common.ExecutionStatus.RUNNING)
        return plugin.function(*args, **kwargs)


def exec_fit(
    model_type: str,
    instance: Any,
    state_cb: common.StateCallback = lambda state: None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Uses a loaded plugin to fit a model instance"""
    with _StateManager(state_cb) as state:
        plugin = decorators.get_fit(model_type)

        args, kwargs = _preprocess_args(args, kwargs, plugin, state)

        args, kwargs = _args_add_instance(
            plugin.instance_arg_name, instance, args, kwargs
        )

        state.set_status(common.ExecutionStatus.RUNNING)
        return plugin.function(*args, **kwargs)


def exec_predict(
    model_type: str,
    instance: Any,
    state_cb: common.StateCallback = lambda state: None,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Uses a loaded plugin to perform a prediction with a given model
    instance. Also returns the instance because it might be altered during the
    prediction, e.g. with reinforcement learning models."""
    with _StateManager(state_cb) as state:
        plugin = decorators.get_predict(model_type)

        args, kwargs = _preprocess_args(args, kwargs, plugin, state)

        args, kwargs = _args_add_instance(
            plugin.instance_arg_name, instance, args, kwargs
        )

        state.set_status(common.ExecutionStatus.RUNNING)
        return instance, plugin.function(*args, **kwargs)


def exec_serialize(model_type: str, instance: Any) -> ByteString:
    """Uses a loaded plugin to convert model into bytes"""
    plugin = decorators.get_serialize(model_type)
    return plugin.function(instance)


def exec_deserialize(model_type: str, instance_bytes: ByteString) -> Any:
    """Uses a loaded plugin to convert bytes into a model instance"""
    plugin = decorators.get_deserialize(model_type)
    return plugin.function(instance_bytes)


class _StateManager:

    def __init__(self, state_cb):
        self._state_cb = state_cb
        self._state = common.ExecutionState(
            status=common.ExecutionStatus.INIT,
            data_access={},
            action=None,
        )

    def get_status(self):
        return self._state.status

    def set_status(self, status: common.ExecutionStatus):
        self._state = self._state._replace(status=status)
        self._state_cb(self._state)

    def update_action(self, value: Any):
        self._state = self._state._replace(action=value)
        self._state_cb(self._state)

    def update_data_access_arg(
        self, arg_name: Union[str, int], substate: common.ExecutionState
    ):
        data_access_state = self._state.data_access
        data_access_state = {**data_access_state, arg_name: substate}
        self._state = self._state._replace(data_access=data_access_state)
        self._state_cb(self._state)

    def __enter__(self):
        self._state_cb(self._state)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.set_status(
            common.ExecutionStatus.ERROR
            if exc_type
            else common.ExecutionStatus.COMPLETE
        )


def _preprocess_args(args, kwargs, plugin, state):
    args, kwargs = _handle_data_access_args(args, kwargs, state)
    kwargs = _kwargs_add_state_cb(
        plugin.state_cb_arg_name,
        state.update_action,
        kwargs,
    )
    return args, kwargs


def _handle_data_access_args(args, kwargs, state: _StateManager):
    updates = {}
    args = list(args)
    for arg_name, arg in list(enumerate(args)) + list(kwargs.items()):
        if not isinstance(arg, common.DataAccessArg):
            continue
        if state.get_status() != common.ExecutionStatus.DATA_ACCESS:
            state.set_status(common.ExecutionStatus.DATA_ACCESS)
        updates[arg_name] = exec_data_access(
            arg.name,
            lambda substate: state.update_data_access_arg(arg_name, substate),
            *arg.args,
            **arg.kwargs,
        )

    for arg_name, dataset in updates.items():
        if isinstance(arg_name, str):
            kwargs[arg_name] = dataset
        elif isinstance(arg_name, int):
            args[arg_name] = dataset
        else:
            raise Exception(f"incorrect arg type {type(arg_name)}")

    return tuple(args), kwargs


def _kwargs_add_state_cb(state_cb_arg_name, cb, kwargs):
    if state_cb_arg_name:
        if state_cb_arg_name in kwargs:
            raise Exception("state cb already set")
        kwargs = dict(kwargs, **{state_cb_arg_name: cb})
    return kwargs


def _args_add_instance(instance_arg_name, instance, args, kwargs):
    if instance_arg_name:
        if instance_arg_name in kwargs:
            raise Exception("instance already set")
        kwargs = dict(kwargs, **{instance_arg_name: instance})
        return args, kwargs
    return (instance, *args), kwargs
