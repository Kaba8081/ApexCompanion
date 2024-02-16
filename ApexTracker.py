import logging as log
import colorama
import time

import pygetwindow as gw
import pytesseract
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
        
        result = func(*args, **kwargs)

        end = time.time()

        log.debug(f"Function '{func.__name__}' took '{str(end-start)[:6]}' seconds to execute.")

        return result
    
    return inner_func

class GameState(Enum):
    LOGIN_SCREEN = 0
    LOBBY = 1
    IN_QUEUE = 2
    IN_DROPSHIP = 4
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

class ApexTracker:
    def __init__(self, config, log_level=log.DEBUG) -> None:
        self.STATE = GameState.LOGIN_SCREEN
        self.CONFIG = config
        self.APEX_MAPS = self.CONFIG["maps"]

        self.is_running = True
        self.current_map = "LOBBY"
        self.recording = True
        self.last_capture = None # last screen capture before death

        self.debug_ignore_focus = True

        log.basicConfig(
            level=log_level,
            #filename='apex_tracker.log', # uncomment to disable console logs
            format='[%(levelname)s] %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S",
            )

    def update(self, action: TrackerControls) -> None:
        match action:
            case TrackerControls.EXIT:
                log.info("Exiting...")
                self.is_running = False
                sys.exit(0)

            case TrackerControls.RECORDING:
                # TODO: fix - when passing trough keyboard module, value does not change
                self.recoding = not self.recording
                log.info(f"Recording: {self.recording}")

            case TrackerControls.DEBUG:
                # used for debugging purposes
                log.debug(tracker.checkGameState())
        return
    
    def checkGameState(self) -> GameState | None:
        # Check the current screen for its state and return the corresponding GameState

        if self.windowIsFocused():
            curr_screen = self.captureScreen()
            
            # Order of checks:
            # KNOCKED
            # IN_GAME
            # DEAD
            # IN_QUEUE
            # LOBBY
            # IN_DROPSHIP
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

        return None

    def captureScreen(self) -> Image.Image | None:
        if (self.windowIsFocused() or self.debug_ignore_focus) and self.recording:
            return ImageGrab.grab()
        return None

    def checkIfObjectOnScreen(self, object: str | list, conf: float=.8, screen: Image.Image=None) -> bool:
        # Find given object on screen and return True if found, False otherwise

        if not screen:
            screen = self.captureScreen()
        
        object = [object] if type(object) == str else object

        for obj in object:
            # convert PIL Image to numpy array
            img_rgb = np.array(screen.convert('RGB')) 
            # convert color space from RGB to GRAY
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            template = cv2.imread(f"{DIR_PATH}\game_assets\{obj}.png", cv2.IMREAD_GRAYSCALE)
            if template is None:
                log.error(f"Error in loading template: {obj}")
                return False
            
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= conf)

            if len(loc[0]) > 0:
                return True
        return False
    
    def saveDeathLocation(self, lastCapture: Image.Image) -> None:
        if self.CONFIG["trackDeaths"]:
            curr_map = self.current_map if self.current_map != "WORLO'S EDGE" else "WORLD'S EDGE"
            save_dir = os.path.join(self.CONFIG["dirDeathCapture"], curr_map)
            last_file = 0

            # check if the directory exists, if not create it
            if os.path.exists(save_dir):
                # if the current map directory exists, get the last death id
                dir_contents = os.listdir(save_dir)
                if dir_contents:
                    last_file = sorted(dir_contents)[-1]
                    last_file = int(last_file.split(".")[0]) 
            else:
                os.makedirs(save_dir, exist_ok=True)

            log.info(f"Saving death location '{save_dir}/{last_file+1}.png'...")
            
            lastCapture.crop((55, 55, 229, 229)).save(f"{save_dir}/{last_file+1}.png")
            
            return
    
    def updateMap(self, screen: Image.Image) -> None:
        img = np.array(screen.convert('RGB')) 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 

        curr_map = pytesseract.image_to_string(img).strip().upper()
        log.debug(f"Map text: {curr_map}")

        if curr_map in self.APEX_MAPS:
            self.current_map = curr_map
            log.info(f"Current map: {self.current_map}") 
        else: 
            log.warning(f"Map name not recognized: {curr_map}")
        
        return
    
    def gameIsRunning(self, process_name: str='r5apex.exe') -> bool | None:    
        return process_name in [p.name() for p in process_iter()]
    
    def windowIsFocused(self, window_name: str='Apex Legends') -> bool | None:
        if self.debug_ignore_focus:
            return True
        elif self.gameIsRunning():
            return gw.getWindowsWithTitle(window_name)[0].isActive
        return False

    @debugAnalyzePerformance
    def devFindOnScreen(
        self, 
        object: str | list, 
        conf: float=.8, 
        screen: Image.Image=None, 
        method=cv2.TM_CCOEFF_NORMED
    ) -> bool:

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
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            cv2.imwrite(f'dev_{obj}.png', img_rgb)

            # if len(loc[0]) > 0:
            #     return True
            # return False

pytesseract.pytesseract.tesseract_cmd = r'E:\przydatne programy\tesseract\tesseract.exe'

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CAPTURE_PATH = os.path.join(DIR_PATH, "captures")
APEX_MAPS = ["LOBBY", "KINGS CANYON", "WORLO'S EDGE", "OLYMPUS", "STORM POINT", "BROKEN MOON"] # ocr mistakes "o" for "d"
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
    "trackDeaths": True,
    "autoQueue": False, # To be implemeneted

    # configuration
    "maps": APEX_MAPS,
    "screenCaptureDelay": .5, # in seconds, delay between screen captures
    "keybinds": KEYBINDS,
    "dirPath": DIR_PATH,
    "dirDeathCapture": CAPTURE_PATH,
}

if __name__ == "__main__":
    tracker = ApexTracker(CONFIG)

    for key in tracker.CONFIG["keybinds"].keys():
        keyboard.add_hotkey(key, tracker.update, args=(tracker.CONFIG["keybinds"][key],))

    if tracker.gameIsRunning() or tracker.debug_ignore_focus:
        while tracker.is_running:
            new_state = tracker.checkGameState()

            if tracker.STATE != new_state and new_state is not None:
                if new_state in [GameState.IN_DROPSHIP, GameState.ALIVE]:
                    tracker.last_capture = None
                    tracker.recording = True

                elif new_state in [GameState.KNOCKED, GameState.DEAD] and CONFIG["trackDeaths"]:
                    if tracker.last_capture:
                        tracker.saveDeathLocation(tracker.last_capture)
                        tracker.last_capture = None

                elif new_state == GameState.IN_QUEUE:
                    tracker.updateMap(tracker.last_capture.crop((54, 859, 326, 896)))

                elif new_state == GameState.LOBBY:
                    tracker.last_capture = tracker.captureScreen()
                    tracker.recording = False

                log.info(f"Changing game state to '{new_state.name}'")
                tracker.STATE = new_state
                
            if tracker.recording:
                time.sleep(CONFIG["screenCaptureDelay"])
            else:
                time.sleep(3)
    else:
        log.error("Apex Legends is not running.")
 
    sys.exit(0)