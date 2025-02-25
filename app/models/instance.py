from typing import Any
import msgspec

class Instance(msgspec.Struct):
    name: str = "Default"

    def __setattr__(self, key: str, value: Any) -> None:
        if hasattr(self, key) and getattr(self, key) == value:
            return
        super().__setattr__(key, value)

    def as_dict(self, skip_private: bool = True) -> dict[str, Any]:
        skip_attributes: list[str] = []

        data = {}

        for key in self.__struct_fields__:
            if key in skip_attributes:
                continue
            if skip_private and key.startswith("_"):
                continue

            data[key] = getattr(self, key)

        return data