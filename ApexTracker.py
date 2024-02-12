import logging as log
import colorama
import time

import pygetwindow as gw
import numpy as np
import keyboard
import sys
import os

from psutil import process_iter
from enum import Enum

from PIL import Image, ImageGrab
import cv2

def debugAnalyzePerformance(func: callable) -> callable:
    def inner_func(*args, **kwargs):
        start = time.time()
        
        func(*args, **kwargs)

        end = time.time()

        log.debug(f"Function '{func.__name__}' took '{str(end-start)[:6]}' seconds to execute.")
    
    return inner_func

class GameState(Enum):
    LOBBY = 1
    IN_QUEUE = 2
    IN_DROPSHIP = 4
    IN_GAME = 4
    ALIVE = 5
    KNOCKED = 6
    DEAD = 7

class TrackerControls(Enum):
    EXIT = 1 # Close application
    RECORDING = 2 # Start/Stop screen recording
    INTERACT = 3 # In-game interact key
    TACTITAL = 4 # Tactical ability
    MOVE_FORWARD = 5 # Movement keys
    MOVE_BACKWARD = 6
    MOVE_LEFT = 7
    MOVE_RIGHT = 8
    DEBUG = 9 # Debugging purposes

class ApexMaps(Enum):
    LOBBY = 1
    KINGS_CANYON = 2
    WORLDS_EDGE = 3
    OLYMPUS = 4
    STORM_POINT = 5
    BROKEN_MOON = 6

class ApexTracker:
    def __init__(self, config, log_level=log.DEBUG) -> None:
        self.STATE = GameState.LOBBY
        self.CONFIG = config

        self.RECORDING = False
        self.CURRENT_MAP = ApexMaps.LOBBY

        self.debug_ignore_focus = True

        log.basicConfig(
            level=log_level,
            #filename='apex_tracker.log', # uncomment to disable console logs
            format='[%(levelname)s] %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S",
            )
        
    def start(self, options: list) -> None:
        while True:
            if self.gameIsRunning():
                self.STATE = GameState.IN_GAME
                break

    def update(self, action: TrackerControls) -> None:
        match action:
            case TrackerControls.EXIT:
                log.info("Exiting...")
                sys.exit(0)
            case TrackerControls.RECORDING:
                if self.STATE == GameState.IN_GAME and self.windowIsFocused():
                    self.RECORDING = not self.RECORDING
                    log.info(f"Recording: {self.RECORDING}")
            case TrackerControls.DEBUG:
                # used for debugging purposes
                log.debug(tracker.checkGameState())
        return
    
    @debugAnalyzePerformance
    def checkGameState(self) -> GameState | None:
        if self.windowIsFocused():
            curr_screen = self.captureScreen()
            
            # Order of checks:
            # KNOCKED
            # IN_GAME
            # DEAD
            # IN_QUEUE
            # LOBBY
            # IN_DROPSHIP
            if self.devFindOnScreen(
                                    ["ig_activate", "ig_bleedingOut"], 
                                    conf=.6, 
                                    screen=curr_screen.crop((53, 907, 1226, 1066))):
                return GameState.KNOCKED
            elif self.devFindOnScreen(
                                    "ig_alive", 
                                    conf=.7, 
                                    screen=curr_screen.crop((1624, 43, 1882, 98))):
                return GameState.ALIVE
            elif self.devFindOnScreen(
                                    "ig_returnToLobby",
                                    conf=.5,
                                    screen=curr_screen.crop((1403, 1016, 1920, 1080)),
                                    method=cv2.TM_SQDIFF_NORMED):
                return GameState.DEAD
            elif self.devFindOnScreen("lb_cancel", screen=curr_screen):
                return GameState.IN_QUEUE
            elif self.devFindOnScreen(
                                    ["lb_fill_teammates", "lb_ready"], 
                                    conf=.7, 
                                    screen=curr_screen.crop((0, 605, 444, 1080))
                                    ):
                return GameState.LOBBY
            elif self.devFindOnScreen(
                                    ["ds_ping", "ds_launch"], 
                                    conf=.7,
                                    screen=curr_screen.crop((737, 771, 1182, 1023))):
                return GameState.IN_DROPSHIP

        return None
    
    def saveDeathLocation(self, lastCapture) -> None:
        if self.CONFIG["trackDeaths"]:
            save_dir = os.path.join(self.CONFIG["dirDeathCapture"], self.CURRENT_MAP.name)
            last_file = 0

            # if the current map directory exists, get the last death id
            dir_contents = os.listdir(save_dir)
            if dir_contents:
                last_file = sorted(dir_contents)[-1]
                last_file = int(last_file.split(".")[0]) 

            log.info("Saving death location...")
            lastCapture.save(f"{save_dir}/{last_file+1}.png")
            
            return

    def captureScreen(self) -> Image.Image | None:
        if self.windowIsFocused() or self.debug_ignore_focus:
            return ImageGrab.grab()
        return None

    def checkIfObjectOnScreen(self, object: str | list, conf: float=.8, screen: Image.Image=None) -> bool:
        if not screen:
            screen = self.captureScreen()
        
        object = [object] if type(object) == str else object

        for obj in object:
            # convert PIL Image to numpy array
            img_rgb = np.array(screen.convert('RGB')) 
            # convert color space from RGB to GRAY
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

            template = cv2.imread(f"{DIR_PATH}\game_assets\{obj}.png", cv2.IMREAD_GRAYSCALE)
            if template is None:
                log.error(f"Error in loading template: {obj}")
                return False
            
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= conf)

            log.debug(f"Found {len(loc[0])} matches for {obj} on screen with {conf} confidence.")

            if len(loc[0]) > 0:
                return True
            return False

    def gameIsRunning(self, process_name: str='r5apex.exe') -> bool | None:    
        return process_name in [p.name() for p in process_iter()]
    
    def windowIsFocused(self, window_name: str='Apex Legends') -> bool | None:
        if self.debug_ignore_focus:
            return True
        elif self.gameIsRunning():
            return gw.getWindowsWithTitle(window_name)[0].isActive
        return False

    def devFindOnScreen(self, object: str | list, conf: float=.8, screen: Image.Image=None, method=cv2.TM_CCOEFF_NORMED) -> bool:
        if not screen:
            screen = self.captureScreen()

        object = [object] if type(object) == str else object

        for obj in object:
            # convert PIL Image to numpy array
            img_rgb = np.array(screen.convert('RGB')) 
            # convert color space from RGB to GRAY
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

            template = cv2.imread(f"{DIR_PATH}\game_assets\{obj}.png", cv2.IMREAD_GRAYSCALE)
            if template is None:
                log.error(f"Error in loading template: {obj}")
                return False
            
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(img_gray, template, method)
            loc = np.where(res >= conf)
            matches = f"{colorama.Fore.RED if len(loc[0]) == 0 else colorama.Fore.GREEN}{len(loc[0])}{colorama.Style.RESET_ALL}"

            log.debug(f"Found {matches} matches for {obj} on screen with {conf} confidence.")

            for pt in zip(*loc[::-1]):
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            cv2.imwrite(f'dev_{obj}.png', img_rgb)

            # if len(loc[0]) > 0:
            #     return True
            # return False

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CAPTURE_PATH = os.path.join(DIR_PATH, "captures")
KEYBINDS = {
    # Companion keybinds
    "m" : TrackerControls.RECORDING, 
    "page down" : TrackerControls.RECORDING, 
    "page up" : TrackerControls.EXIT,

    # Apex Legends in-game controlls
    "e": TrackerControls.INTERACT,

    # Debug keybinds
    "end": TrackerControls.DEBUG,
}
CONFIG = {
    # Companion features
    "trackDeaths": False,
    "autoQueue": False, # To be implemeneted

    # configuration
    "screenCaptureDelay": 0.5, # in seconds, delay between screen captures
    "keybinds": KEYBINDS,
    "dirPath": DIR_PATH,
    "dirDeathCapture": CAPTURE_PATH,
}

if __name__ == "__main__":
    tracker = ApexTracker(CONFIG)

    #tracker.devFindOnScreen("fill_teammates", conf=0.7, screen=Image.open(f"{DIR_PATH}/dev_assets/apexLobby.png"))
    #tracker.devFindOnScreen("fill_teammates", conf=0.7)

    if tracker.gameIsRunning():
        while True:
            event = keyboard.read_event()

            if event.event_type == keyboard.KEY_DOWN and event.name in CONFIG["keybinds"].keys():
                tracker.update(CONFIG["keybinds"][event.name])
    else:
        log.error("Apex Legends is not running.")
 
    sys.exit(0)