from llama_index.core.constants import DATA_KEY, TYPE_KEY
from llama_index.core.data_structs.data_structs import IndexStruct
from llama_index.core.data_structs.registry import (
    INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS,
)


def index_struct_to_json(index_struct: IndexStruct) -> dict:
    """Serialize an index struct into a JSON-compatible dict, tagging it with its type."""
    return {
        TYPE_KEY: index_struct.get_type(),
        DATA_KEY: index_struct.to_json(),
    }


def json_to_index_struct(struct_dict: dict) -> IndexStruct:
    """
    Deserialize a dict produced by `index_struct_to_json` back into an `IndexStruct`.

    Looks up the concrete `IndexStruct` subclass from the type tag and
    falls back to `from_dict` if the class does not support `from_json`.
    """
    type = struct_dict[TYPE_KEY]
    data_dict = struct_dict[DATA_KEY]
    cls = INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS[type]
    try:
        return cls.from_json(data_dict)
    except TypeError:
        return cls.from_dict(data_dict)
