import base64
import itertools
from typing import Any

from hat import aio
from hat import util
import hat.event.common

from aimm.server import common
from aimm import plugins
from aimm.server.common import Model


def create_subscription(conf):
    return [tuple([*conf["model_prefix"], "*"])]


async def create(conf, event_client):
    common.json_schema_validator.validate(
        "aimm://server/backend/event.yaml#", conf
    )
    backend = EventBackend(conf, event_client)
    await backend.start()

    return backend


class EventBackend(common.Backend):
    def __init__(self, conf, event_client):
        self._model_prefix = conf["model_prefix"]
        self._executor = aio.create_executor()
        self._cbs = util.CallbackRegistry()
        self._group = aio.Group()
        self._client = event_client
        self._id_counter = None

    @property
    def async_group(self) -> aio.Group:
        """Async group"""
        return self._group

    async def start(self):
        models = await self.scan_models()
        self._id_counter = itertools.count(
            max((model.instance_id for model in models), default=1)
        )

    async def scan_models(self) -> list[Model]:
        query_result = await self._client.query(
            hat.event.common.QueryLatestParams(
                event_types=[(*self._model_prefix, "*")]
            )
        )
        return [self._event_to_model(e) for e in query_result.events]

    async def get_instance(self, instance_id: int) -> (str, Any):
        query_result = await self._client.query(
            hat.event.common.QueryLatestParams(
                event_types=[(*self._model_prefix, f"{instance_id}")]
            )
        )

        if not query_result.events:
            raise ValueError("ID not found")
        model_event = query_result.events[0]
        return await self._event_to_instance(model_event)

    async def create_model(self, model_type, instance):
        model = common.Model(
            model_type=model_type,
            instance_id=next(self._id_counter),
        )
        await self._register_model(model, instance)
        return model

    async def update_model(self, model, instance):
        await self._register_model(model, instance)

    def register_model_change_cb(self, cb):
        self._cbs.register(cb)

    async def process_events(self, events):
        for event in events:
            self._cbs.notify(self._event_to_model(event))

    async def _register_model(self, model, instance):
        await self._client.register(
            [await self._model_to_event(model, instance)]
        )

    async def _model_to_event(self, model, instance):
        instance_b64 = base64.b64encode(
            await self._executor(
                plugins.exec_serialize, model.model_type, instance
            )
        ).decode("utf-8")
        return hat.event.common.RegisterEvent(
            type=(*self._model_prefix, str(model.instance_id)),
            source_timestamp=None,
            payload=hat.event.common.EventPayloadJson(
                {"type": model.model_type, "instance": instance_b64},
            ),
        )

    def _event_to_model(self, event):
        return common.Model(
            instance_id=int(event.type[len(self._model_prefix)]),
            model_type=event.payload.data["type"],
        )

    async def _event_to_instance(self, event):
        return await self._executor(
            plugins.exec_deserialize,
            event.payload.data["type"],
            base64.b64decode(event.payload.data["instance"].encode("utf-8")),
        )
