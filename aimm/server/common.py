from typing import (
    Any,
    Dict,
    Callable,
    List,
    Tuple,
    NamedTuple,
    Optional,
    Collection,
)
import abc
import enum
import logging

from hat import aio
from hat import util
from hat import json
import hat.event.eventer.client
import hat.event.common

from aimm import plugins
import aimm.common

mlog = logging.getLogger(__name__)

json_schema_validator = json.RsSchemaValidator(aimm.common.json_schema_repo)


CreateSubscription = Callable[[Dict], hat.event.common.Subscription]
CreateSubscription.__doc__ = """
Type of the ``create_subscription`` function that the dynamically imported
controls and backends may implement. Receives component configuration as the
only argument and returns a subscription object.
"""


class Model(NamedTuple):
    """Model descriptor, contains all data necessary to identify and perform
    other actions with it."""

    model_type: str
    """model type, used to identify which plugin to use"""
    instance_id: int
    """instance id"""


class Engine(aio.Resource, abc.ABC):
    """Engine interface"""

    @abc.abstractmethod
    def create_instance(
        self, model_type: str, *args: Any, **kwargs: Any
    ) -> "Action":
        """Starts an action that creates a model instance and stores it in
        state. Action result is the model ID.

        Args:
            model_type: model type
            *args: instantiation arguments
            **kwargs: instantiation keyword arguments"""

    @abc.abstractmethod
    async def scan_models(self) -> List[Model]:
        """Fetch all stored model representations.

        Returns:
            persisted models"""

    @abc.abstractmethod
    async def add_instance(self, model_type: str, instance: Any) -> int:
        """Adds existing instance to the state

        Returns:
            Model ID"""

    @abc.abstractmethod
    async def update_instance(self, model: Model, instance: Any):
        """Update existing instance in the state"""

    @abc.abstractmethod
    def fit(self, instance_id: int, *args: Any, **kwargs: Any) -> "Action":
        """Starts an action that fits an existing model instance. The used
        fitting function is the one assigned to the model type. The instance,
        while it is being fitted, is not accessible by any of the other
        functions that would use it (other calls to fit, predictions, etc.).

        Args:
            instance_id: id of model instance that will be fitted
            *args: arguments to pass to the fitting function - if of type
                :class:`aimm.server.common.DataAccess`, the value passed to the
                fitting function is the result of the call to that plugin,
                other arguments are passed directly
            **kwargs: keyword arguments, work the same as the positional
                arguments"""

    @abc.abstractmethod
    def predict(self, instance_id: int, *args: Any, **kwargs: Any) -> "Action":
        """Starts an action that uses an existing model instance to perform a
        prediction. The used prediction function is the one assigned to model's
        type. The instance, while prediction is called, is not accessible by
        any of the other functions that would use it (other calls to predict,
        fittings, etc.).  If instance has changed while predicting, it is
        updated in the state and database.

        Args:
            instance_id: id of the model instance used for prediction
            *args: arguments to pass to the predict function - if of type
                :class:`aimm.server.common.DataAccess`, the value passed to the
                predict function is the result of the call to that plugin,
                other arguments are passed directly
            **kwargs: keyword arguments, work the same as the positional
                arguments

        Returns:
            Reference to task of the manageable predict call, result of it is
            the model's prediction"""


class ActionStatus(enum.StrEnum):
    INIT = "init"
    RUNNING = "running"
    STORING = "storing"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


class ActionState(NamedTuple):
    """Object representing the state of an individual action"""

    meta: dict[str, Any]
    """Information about the action call"""
    status: ActionStatus
    """Current action status"""
    run: Optional[plugins.ExecutionState]


class Action(aio.Resource, abc.ABC):
    """Represents a manageable call. Is an :class:`aio.Resource` so call can be
    cancelled using ``async_close``."""

    @abc.abstractmethod
    def subscribe_to_state_change(
        self, state_cb: Callable[[ActionState], None]
    ) -> util.RegisterCallbackHandle:
        """Provide a function to be called every time action state changes.

        Args:
            state_cb: function called whenever action state changes"""

    @abc.abstractmethod
    async def wait_result(self) -> Any:
        """Wait until call returns a result. May raise
        :class:`asyncio.CancelledError` in case the call was cancelled."""


def create_subscription(conf: Any) -> list[hat.event.common.EventType]:
    """Placeholder of the backends and controls optional create subscription
    function, needs to satisfy the given signature"""


def create_backend(
    conf: Dict, event_client: Optional[hat.event.eventer.client.Client] = None
) -> "Backend":
    """Placeholder of the backend's create function, needs to satisfy the given
    signature"""


class Backend(aio.Resource, abc.ABC):
    """Backend interface. In order to integrate in the aimm server, create a
    module with the implementation and function ``create`` that creates a
    backend instance. The function should have a signature as the
    :func:`create_backend` function.

    The ``event_client`` argument is not ``None`` if backend module also
    contains function named ``create_subscription`` with the same signature as
    the :func:`create_subscription`. The function receives the same backend
    configuration the ``create`` function would receive and returns the
    subscription object for the backend.
    """

    @abc.abstractmethod
    async def scan_models(self) -> List[Model]:
        """Get all persisted models.

        Returns:
            persisted models"""

    @abc.abstractmethod
    async def get_instance(self, instance_id: int) -> Tuple[str, Any]:
        """Get deserialized model instance, requires that a deserialization
        function is defined for the persisted type

        Returns:
            Model type and instance."""

    async def update_instance(self, model: Model, instance: Any):
        """Set model instance, requires a serialization function configured for
        that type.

        Args:
            model: model structure containing model identification info
            instance: new model instance to be stored
        """

    @abc.abstractmethod
    async def create_model(self, model_type: str, instance: Any) -> int:
        """Store a new model, requires that a serialization for the model type
        is defined

        Returns:
            Model ID."""

    @abc.abstractmethod
    async def update_model(self, model: Model, instance: Any):
        """Replaces the old stored model instance with the new one, requires
        that a serialization is defined for the model type"""

    async def process_events(self, events: hat.event.common.Event):
        """Implementation optional. Called when event client receives events
        matched by subscription from `create_backend_subscription`. Ignores
        events by default, with a warning log."""
        mlog.warning(
            "received events when no process_event method was implemented"
        )


def create_control(
    conf: Dict,
    engine: Engine,
    event_client: Optional[hat.event.eventer.client.Client] = None,
) -> "Control":
    """Placeholder of the control's create function, needs to satisfy the given
    signature"""


class Control(aio.Resource, abc.ABC):
    """Control interface. In order to integrate in the aimm server, create a
    module with the implementation and function ``create`` that creates a
    control instance and should have a signature as the :func:`create_control`
    function.

    The ``event_client`` argument is not ``None`` if control module also
    contains function named ``create_subscription`` with the same signature as
    the :func:`create_subscription`.  The function receives the same control
    configuration the ``create`` function would receive and returns the list of
    subscriptions ``ProxyClient`` should subscribe to.
    """

    async def process_events(self, events: Collection[hat.event.common.Event]):
        """Implementation optional. Called when event client receives events
        matched by subscription from `create_backend_subscription`. Ignores
        events by default, with a warning log."""
        mlog.warning(
            "received events when no process_event method was implemented"
        )
