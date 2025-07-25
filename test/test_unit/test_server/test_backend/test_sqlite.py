import pytest

from aimm.server.backend import sqlite
from aimm.server import common
from aimm import plugins


@pytest.fixture
async def backend(tmp_path):
    backend = await sqlite.create({"path": str(tmp_path / "backend.db")}, None)
    yield backend
    await backend.async_close()


async def test_create(tmp_path):
    backend_object = await sqlite.create(
        {"path": str(tmp_path / "backend.db")}, None
    )
    assert backend_object
    await backend_object.async_close()


async def test_models(backend, plugin_teardown):
    @plugins.serialize(["test"])
    def serialize(instance_object):
        return instance_object.encode("utf-8")

    @plugins.deserialize(["test"])
    def deserialize(instance_blob):
        return instance_blob.decode("utf-8")

    await backend.create_model("test", "instance")
    expected_model = common.Model(
        instance_id=1, model_type="test"
    )
    assert await backend.scan_models() == [expected_model]

    await backend.update_model(expected_model, instance="instance")
    instance = await backend.get_instance(1)
    assert instance == "instance"
