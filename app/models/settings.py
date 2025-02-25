from typing import Any, Dict
import json

from app.models.instance import Instance
from app.utils.app_info import AppInfo

class Settings:
    def __init__(self) -> None:
        self._settings_file = AppInfo().app_settings_file

        self.instances: Dict[str, Instance] = {"Default": Instance()}

    def __setattr(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            super().__setattr__(key, value)
            return

        if hasattr(self, key) and getattr(self, key) == value:
            return

        super().__setattr__(key, value)

    def load(self) -> None:
        try:
            with open(str(self._settings_file), "r") as file:
                data = json.loads(file)
        except FileNotFoundError:
            self.save()
        except json.JSONDecodeError:
            raise

    def save(self) -> None:
        with open(str(self._settings_file), "w") as file:
            json.dump(self._to_dict(), file)
    
    def _to_dict(self, skip_private: bool = True) -> Dict[str, Any]:
        special_attributes = ["instances"]
        skip_attributes = ["destroyed", "objectNameChanged"]

        data = {}

        for key, value in self.__dict__.items():
            if key in special_attributes:
                continue
            if key in skip_attributes:
                continue
            if skip_private and key.startswith("_"):
                continue

            data[key] = value
        
        data["instances"] = {
            name: instance.as_dict for name, instance in self.instances.items()
        }