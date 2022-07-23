from enum import Enum
from enum import EnumMeta
from typing import Any


class MetaEnum(EnumMeta):
    def __contains__(cls, item: Any) -> bool:
        return (
            super().__contains__(item)
            if isinstance(item, Enum)
            else item in cls._member_map_.values()
        )


class StringEnum(str, Enum, metaclass=MetaEnum):
    pass
