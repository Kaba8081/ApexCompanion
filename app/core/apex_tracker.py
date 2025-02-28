import logging as log
import colorama
import os
import re

from app.models.settings import Settings
from app.utils.constants import TrackerControls, GameState
from app.utils.app_info import AppInfo

import pygetwindow as gw
from psutil import process_iter
import numpy as np
import pytesseract

from PIL import Image, ImageGrab
import cv2

# TODO: saveDeathLocation
# TODO: rewrite checkIfObjectOnScreen

class ApexTracker:
    def __init__(self, settings: Settings):
        self._settings = settings

        self._game_state = self._settings.tracker_game_states.LOGIN_SCREEN
        self._game_current_map = "LOBBY"

        self._is_recording = True
        self._is_debug = self._settings.app_debug
        
        self._last_capture = None

    def update(self, action: TrackerControls) -> None:
        match action:
            case TrackerControls.EXIT: # TODO: Add a destructor to save settings etc.
                self._is_recording = False
            
            case TrackerControls.RECORDING:
                self.toggleRecording()
        return
    
    def captureScreen(self) -> Image.Image | None:
        """Capture the current screen and return the PIL Image object."""

        if (self.windowIsFocused() and self._is_recording) or self._settings.tracker_ignore_focus:
            return ImageGrab.grab()
        
        return None

    def checkGameState(self) -> GameState | None:
        curr_screen = self.captureScreen()

        # Error occured during capture / game is minimized
        if not curr_screen: 
            return None

        if self.checkIfObjectOnScreen(
                                ["ig_activate", "ig_bleedingOut"], 
                                conf=.6, 
                                screen=curr_screen.crop((53, 907, 1226, 1066))):
            return GameState.KNOCKED
        elif self.checkIfObjectOnScreen(
                                "ig_alive", 
                                conf=.7, 
                                screen=curr_screen.crop((1624, 43, 1882, 98))):
            self.last_capture = curr_screen
            return GameState.ALIVE
        elif self.checkIfObjectOnScreen(
                                "ig_returnToLobby",
                                conf=.85,
                                screen=curr_screen.crop((1403, 1016, 1920, 1080))):
            return GameState.DEAD
        elif self.checkIfObjectOnScreen("lb_cancel", screen=curr_screen):
            return GameState.IN_QUEUE
        elif self.checkIfObjectOnScreen(
                                ["lb_fill_teammates", "lb_ready"], 
                                conf=.7, 
                                screen=curr_screen.crop((0, 605, 444, 1080))
                                ):
            return GameState.LOBBY
        elif self.checkIfObjectOnScreen(
                                ["ds_ping", "ds_launch"], 
                                conf=.7,
                                screen=curr_screen.crop((737, 771, 1182, 1023))):
            return GameState.IN_DROPSHIP

    def checkIfObjectOnScreen(self, to_find: str | list, conf: float = .8, screen: Image.Image=None) -> bool:
        if not screen:
            screen = self.captureScreen()
        
        to_find = [to_find] if type(to_find) == str else to_find
        path = AppInfo().app_assets_folder / "objects"

        for obj in to_find:
            # convert PIL Image to numpy array
            img_rgb = np.array(screen.convert('RGB'))
            # convert RGB TO GRAYSCALE
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            try:
                template = cv2.imread(os.path.join(path, f"{obj}.png"), 0)
            except FileNotFoundError:
                log.error(f"Template not found: {obj}")
                continue
            except Exception as e:
                log.error(f"Error occured when loading template '{obj}.png': {e}")
                continue

            if template is None:
                log.error(f"Template '{obj}.png' is empty.")
                continue
            
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= conf)

            if len(loc[0]) > 0:
                return True
        
        return False

    def saveDeathLocation(self, lastCapture: Image.Image) -> None:
        def atoi(text):
            return int(text) if text.isdigit() else text
        def natural_keys(text):
            return [atoi(c) for c in re.split(r'(\d+)', text)]

        if self._settings.tracker_track_deaths:
            curr_map = self._game_current_map

            save_dir = AppInfo().app_captures_folder / curr_map
            last_file = 0

            if not os.path.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            
            dir_contents = os.listdir(save_dir)
            dir_contents.sort(key=natural_keys)

            if dir_contents:
                last_file = int(dir_contents[-1].split(".")[0])
            
            log.info("Saving death location: %s/%d", save_dir, last_file)

            # crop only the minimap from the screen
            # TODO: Adjuct crop dimensions based on user's resolution
            lastCapture.crop((55, 55, 229, 229)).save(os.path.join(save_dir), f"{last_file+1}.png")

        return

    def updateMap(self, screen: Image.Image) -> None:
        img = np.array(screen)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        curr_map = pytesseract.image_to_string(img).strip().upper()
        log.debug("Map detected: %s", curr_map)

        # TODO: make a function to check if the name isn't misspelled
        if curr_map in self._settings.tracker_maps:
            self._game_current_map = curr_map
            log.info("Current map: %s", curr_map)
        else:
            log.warning(f"Map name not recognized: %s", curr_map)

    def toggleRecording(self) -> None:
        self._is_recording = not self._is_recording
        log.info("Recording: %s%s%s", colorama.Fore.GREEN if self.recording else colorama.Fore.RED, self.recording, colorama.Style.RESET_ALL)

    @property
    def gameIsRunning(self) -> bool:
        return 'r5apex.exe' in [p.name() for p in process_iter()]

    @property
    def windowIsFocused(self) -> bool:
        if self.gameIsRunning():
            return gw.getWindowsWithTitle('Apex Legends')[0].isActive
        return False