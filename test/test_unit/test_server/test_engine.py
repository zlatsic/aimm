from typing import Any, Dict, Tuple

from hat import aio
import pytest

from aimm.server import engine
from aimm.server import common
from aimm import plugins


class MockBackend(common.Backend):
    def __init__(self, instances: Dict[int, Tuple[str, Any]] | None = None):
        self._group = aio.Group()
        # maps instance_id -> (model_type, instance)
        self._instances: Dict[int, Tuple[str, Any]] = instances or {}

    @property
    def async_group(self):
        return self._group

    async def scan_models(self):
        return [
            common.Model(model_type=model_type, instance_id=instance_id)
            for instance_id, (model_type, _instance) in self._instances.items()
        ]

    async def get_instance(self, instance_id: int):
        return self._instances[instance_id]

    async def create_model(self, model_type, instance):
        instance_id = max(self._instances.keys(), default=0) + 1
        self._instances[instance_id] = (model_type, instance)
        return instance_id

    async def update_model(self, model, instance):
        # keep for interface completeness
        self._instances[model.instance_id] = (model.model_type, instance)

    async def update_instance(self, model, instance):
        self._instances[model.instance_id] = (model.model_type, instance)


def create_engine(backend=None):
    backend = backend or MockBackend()
    return engine.Engine(
        {
            "sigterm_timeout": 1,
            "max_children": 1,
            "check_children_period": 0.2,
        },
        backend,
    )


async def test_empty():
    eng = create_engine()
    assert eng is not None
    assert isinstance(eng, common.Engine)
    await eng.async_close()


async def test_models_in_backend():
    instances = {1: ("test", None), 2: ("test", None)}
    eng = create_engine(MockBackend(instances))
    models_eng = await eng.scan_models()
    assert set(models_eng) == {
        common.Model(model_type="test", instance_id=1),
        common.Model(model_type="test", instance_id=2),
    }
    await eng.async_close()


@pytest.mark.timeout(2)
async def test_create_instance(plugin_teardown):
    backend = MockBackend()
    eng = create_engine(backend)

    @plugins.instantiate("test")
    def create(*c_args, **c_kwargs):
        return "test", c_args, c_kwargs

    args = (1, 2, 3)
    kwargs = {"p1": 4, "p2": 5}
    action = eng.create_instance("test", *args, **kwargs)
    states = []
    action.subscribe_to_state_change(lambda s: states.append(s))

    model_id = await action.wait_result()
    assert model_id == 1
    # instance stored in backend
    assert backend._instances[1] == ("test", ("test", args, kwargs))

    # Verify complete state transitions
    assert len(states) == 7  # INIT, RUNNING (4x), STORING, COMPLETE

    # Verify state transition order
    statuses = [state.status for state in states]
    assert statuses[0] == common.ActionStatus.INIT
    assert all(
        status == common.ActionStatus.RUNNING for status in statuses[1:5]
    )
    assert statuses[5] == common.ActionStatus.STORING
    assert statuses[6] == common.ActionStatus.COMPLETE

    # Check INIT state
    init_state = states[0]
    assert init_state.status == common.ActionStatus.INIT
    assert init_state.meta == {
        "call": "create_instance",
        "model_type": "test",
        "args": ["1", "2", "3"],
        "kwargs": {"p1": "4", "p2": "5"},
    }
    assert init_state.run is None

    # Check RUNNING states (multiple due to plugin execution)
    for i in range(1, 5):
        running_state = states[i]
        assert running_state.status == common.ActionStatus.RUNNING
        assert running_state.meta == init_state.meta
        # Note: run field may be None if plugin execution doesn't provide
        # execution state

    # Check STORING state
    storing_state = states[5]
    assert storing_state.status == common.ActionStatus.STORING
    assert storing_state.meta == init_state.meta
    assert storing_state.run is not None

    # Check COMPLETE state
    complete_state = states[6]
    assert complete_state.status == common.ActionStatus.COMPLETE
    assert complete_state.meta == init_state.meta
    assert complete_state.run is not None
    await eng.async_close()


@pytest.mark.timeout(2)
async def test_add_instance(plugin_teardown):
    backend = MockBackend()
    eng = create_engine(backend)
    instance_id = await eng.add_instance("test", None)
    assert instance_id == 1
    assert backend._instances[1] == ("test", None)
    await eng.async_close()


@pytest.mark.timeout(2)
async def test_fit(plugin_teardown):
    backend = MockBackend()
    eng = create_engine(backend)

    instance_id = await eng.add_instance("test", "instance")

    @plugins.fit(["test"])
    def fit(*f_args, **f_kwargs):
        return "instance_fitted", f_args, f_kwargs

    args = (1, 2)
    kwargs = {"p1": 3, "p2": 4}

    action = eng.fit(instance_id, *args, **kwargs)
    states = []
    action.subscribe_to_state_change(lambda s: states.append(s))

    result = await action.wait_result()
    assert result is None
    # backend instance should be updated
    assert backend._instances[instance_id] == (
        "test",
        ("instance_fitted", ("instance", *args), kwargs),
    )

    # Verify complete state transitions
    assert len(states) == 7  # INIT, RUNNING (4x), STORING, COMPLETE

    # Check INIT state
    init_state = states[0]
    assert init_state.status == common.ActionStatus.INIT
    assert init_state.meta == {
        "call": "fit",
        "model": instance_id,
        "args": ["1", "2"],
        "kwargs": {"p1": "3", "p2": "4"},
    }
    assert init_state.run is None

    # Check RUNNING states (multiple due to plugin execution)
    for i in range(1, 5):
        running_state = states[i]
        assert running_state.status == common.ActionStatus.RUNNING
        assert running_state.meta == init_state.meta
        # Note: run field may be None if plugin execution doesn't provide
        # execution state

    # Check STORING state
    storing_state = states[5]
    assert storing_state.status == common.ActionStatus.STORING
    assert storing_state.meta == init_state.meta
    assert storing_state.run is not None

    # Check COMPLETE state
    complete_state = states[6]
    assert complete_state.status == common.ActionStatus.COMPLETE
    assert complete_state.meta == init_state.meta
    assert complete_state.run is not None
    await eng.async_close()


async def test_predict(plugin_teardown):
    backend = MockBackend()
    eng = create_engine(backend)

    instance_id = await eng.add_instance("test", ["instance"])

    @plugins.predict(["test"])
    def predict(instance, *p_args, **p_kwargs):
        instance.append(1)
        return instance, p_args, p_kwargs

    args = (1, 2)
    kwargs = {"p1": 3, "p2": 4}

    action = eng.predict(instance_id, *args, **kwargs)
    states = []
    action.subscribe_to_state_change(lambda s: states.append(s))

    expected_prediction = (["instance", 1], args, kwargs)
    prediction = await action.wait_result()
    assert prediction == expected_prediction
    # backend instance should be updated
    assert backend._instances[instance_id] == ("test", ["instance", 1])

    # Verify complete state transitions
    assert len(states) == 7  # INIT, RUNNING (4x), STORING, COMPLETE

    # Check INIT state
    init_state = states[0]
    assert init_state.status == common.ActionStatus.INIT
    assert init_state.meta == {
        "call": "predict",
        "model": instance_id,
        "args": ["1", "2"],
        "kwargs": {"p1": "3", "p2": "4"},
    }
    assert init_state.run is None

    # Check RUNNING states (multiple due to plugin execution)
    for i in range(1, 5):
        running_state = states[i]
        assert running_state.status == common.ActionStatus.RUNNING
        assert running_state.meta == init_state.meta
        # Note: run field may be None if plugin execution doesn't provide
        # execution state

    # Check STORING state
    storing_state = states[5]
    assert storing_state.status == common.ActionStatus.STORING
    assert storing_state.meta == init_state.meta
    assert storing_state.run is not None

    # Check COMPLETE state
    complete_state = states[6]
    assert complete_state.status == common.ActionStatus.COMPLETE
    assert complete_state.meta == init_state.meta
    assert complete_state.run is not None
    await eng.async_close()
