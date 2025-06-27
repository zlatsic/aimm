from hat import json
import hat.monitor.common
import logging
import typing
import importlib.resources


mlog = logging.getLogger(__name__)

with importlib.resources.as_file(
    importlib.resources.files(__package__) / "json_schema_repo.json"
) as _path:
    json_schema_repo: json.SchemaRepository = json.merge_schema_repositories(
        json.json_schema_repo,
        json.decode_file(_path),
        hat.monitor.common.json_schema_repo,
    )
    """JSON schema repository"""


JSON = typing.Union[
    None, bool, int, float, str, typing.List["JSON"], typing.Dict[str, "JSON"]
]
"""JSON serializable data"""
