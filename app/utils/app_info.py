import os
import sys
from pathlib import Path
from platformdirs import PlatformDirs

class AppInfo:

    _instance: "None | AppInfo" = None

    def __new__(cls) -> 'AppInfo':
        if not cls._instance:
            cls.instance = super(AppInfo, cls).__new__(cls)
        
        return cls._instance

    def __init__(self):

        if hasattr(self, '_is_initialized') and self._is_initialized:
            return

        self._app_name = 'Apex Companion'
        self._app_copyright = ''
        self._app_version = '1.0.0'

        platform_dirs = PlatformDirs(appname=self._app_name)
        self._app_storage_folder: Path = Path(platform_dirs.user_data_dir)
        self._user_log_folder: Path = Path(platform_dirs.user_log_dir)

        self._settings_file: Path = self._app_storage_folder / 'settings.json'

        self._app_storage_folder.mkdir(parents=True, exist_ok=True)
        self._user_log_folder.mkdir(parents=True, exist_ok=True)

        self._is_initialized: bool = True

    @property
    def app_name(self) -> str:
        return self._app_name
    
    @property
    def app_version(self) -> str:
        return self._app_version
    
    @property
    def app_copyright(self) -> str:
        return self._app_copyright
    
    @property
    def application_folder(self) -> Path:
        return self._app_storage_folder
    
    @property
    def app_storage_folder(self) -> Path:
        return self._app_storage_folder
    
    @property
    def app_settings_file(self) -> Path:
        return self._settings_file
    
    @property
    def user_log_folder(self) -> Path:
        return self._user_log_folder