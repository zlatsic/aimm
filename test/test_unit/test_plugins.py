from aimm import plugins


class StateMock:
    def __init__(self):
        self.history = []

    def state_cb(self, state):
        self.history.append(state)


def test_instantiate(plugin_teardown):
    @plugins.instantiate("test", state_cb_arg_name="state_cb")
    def instantiate(state_cb):
        state_cb("state update")
        return "instance"

    state = StateMock()
    assert plugins.exec_instantiate("test", state.state_cb) == "instance"
    assert state.history == [
        {"status": "init"},
        {"status": "running"},
        {"status": "running", "action": "state update"},
        {"status": "complete", "action": "state update"},
    ]


def test_data_access(plugin_teardown):
    @plugins.data_access("test", state_cb_arg_name="state_cb")
    def data_access(state_cb):
        state_cb("state update")
        return "data"

    state = StateMock()
    assert plugins.exec_data_access("test", state.state_cb) == "data"
    assert state.history == [
        {"status": "init"},
        {"status": "running"},
        {"status": "running", "action": "state update"},
        {"status": "complete", "action": "state update"},
    ]


def test_fit(plugin_teardown):
    @plugins.fit(
        ["test"], state_cb_arg_name="state_cb", instance_arg_name="instance"
    )
    def fit(state_cb, instance):
        state_cb("state update")
        return instance + "-fit"

    state = StateMock()
    assert (
        plugins.exec_fit("test", "instance", state.state_cb) == "instance-fit"
    )

    assert state.history == [
        {"status": "init"},
        {"status": "running"},
        {"status": "running", "action": "state update"},
        {"status": "complete", "action": "state update"},
    ]


def test_predict(plugin_teardown):
    @plugins.predict(
        ["test"], state_cb_arg_name="state_cb", instance_arg_name="instance"
    )
    def predict(state_cb, instance):
        state_cb("state update")
        return "predict-result"

    state = StateMock()
    assert plugins.exec_predict("test", "instance", state.state_cb) == (
        "instance",
        "predict-result",
    )
    assert state.history == [
        {"status": "init"},
        {"status": "running"},
        {"status": "running", "action": "state update"},
        {"status": "complete", "action": "state update"},
    ]


def test_serialize(plugin_teardown):
    @plugins.serialize(["test"])
    def serialize(instance):
        return instance

    assert plugins.exec_serialize("test", "instance") == "instance"


def test_deserialize(plugin_teardown):
    @plugins.deserialize(["test"])
    def deserialize(i_bytes):
        return i_bytes

    instance_bytes = "instance bytes".encode("utf-8")

    assert plugins.exec_deserialize("test", instance_bytes) == instance_bytes


def test_model(plugin_teardown):
    @plugins.model
    class Model1(plugins.Model):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.fit_args = []
            self.fit_kwargs = {}

        def fit(self, *args, **kwargs):
            self.fit_args = args
            self.fit_kwargs = kwargs
            return self

        def predict(self, *args, **kwargs):
            return args, kwargs

        def serialize(self):
            return bytes()

        @classmethod
        def deserialize(cls, _):
            return Model1()

    model_type = "test_plugins.Model1"

    state = StateMock()
    model = plugins.exec_instantiate(
        model_type, state.state_cb, "a1", "a2", k1="1", k2="2"
    )
    assert model.args == ("a1", "a2")
    assert model.kwargs == {"k1": "1", "k2": "2"}
    assert state.history == [
        {"status": "init"},
        {"status": "running"},
        {"status": "complete"},
    ]

    state = StateMock()
    plugins.exec_fit(model_type, model, state.state_cb, "fit_a1", fit_k1="1")
    assert model.fit_args == ("fit_a1",)
    assert model.fit_kwargs == {"fit_k1": "1"}
    assert state.history == [
        {"status": "init"},
        {"status": "running"},
        {"status": "complete"},
    ]
