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

        # Directories
        main_file = sys.modules["__main__"].__file__
        if main_file is None:
            raise Exception('Unable to get the main file path.')
        
        self._application_folder = Path(
            Path(main_file).resolve().parent
            if "__compiled__" in globals()
            else Path(__file__).resolve().parent.parent
        )

        platform_dirs = PlatformDirs(appname=self._app_name)
        self._app_storage_folder: Path = Path(platform_dirs.user_data_dir)
        self._user_log_folder: Path = Path(platform_dirs.user_log_dir)

        self._settings_file: Path = self._app_storage_folder / 'settings.json'
        self._app_captures_folder: Path = self._application_folder / 'captures'
        self._app_result_folder: Path = self._application_folder / 'results'

        self._app_storage_folder.mkdir(parents=True, exist_ok=True)
        self._user_log_folder.mkdir(parents=True, exist_ok=True)
        self._app_captures_folder.mkdir(parents=True, exist_ok=True)
        self._app_result_folder.mkdir(parents=True, exist_ok=True)

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

    #region Directories
    @property
    def application_folder(self) -> Path:
        return self._app_storage_folder
    
    @property
    def app_storage_folder(self) -> Path:
        return self._app_storage_folder

    @property
    def user_log_folder(self) -> Path:
        return self._user_log_folder
    
    @property
    def app_captures_folder(self) -> Path:
        return self._app_captures_folder

    @property
    def app_result_folder(self) -> Path:
        return self._app_result_folder
    #endregion

    #region File Directories
    @property
    def app_settings_file(self) -> Path:
        return self._settings_file
    #endregion