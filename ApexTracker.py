import logging as log

from enum import Enum
import pygetwindow as gw
import keyboard
import sys
import os

from psutil import process_iter

class GameState(Enum):
    LOBBY = 0
    IN_DROPSHIP = 1
    IN_GAME = 2

class TrackerControls(Enum):
    EXIT = 0 # Close application
    RECORDING = 1 # Start/Stop screen recording

class ApexTracker:
    def __init__(self, config, log_level=log.DEBUG) -> None:
        self.STATE = GameState.LOBBY
        self.CONFIG = config

        self.RECORDING = False

        log.basicConfig(
            level=log_level,
            #filename='apex_tracker.log', # uncomment to disable console logs
            format='[%(asctime)s][%(levelname)s] %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S",
            )
        
    def startTracker(self, options: list):
        pass

    def update(self, action: TrackerControls) -> None:
        if action == TrackerControls.EXIT:
            log.info("Exiting...")
            sys.exit(0)
            
        elif action == TrackerControls.RECORDING:
            if self.STATE == GameState.IN_GAME and self.windowIsFocused():
                self.RECORDING = not self.RECORDING
                log.info(f"Recording: {self.RECORDING}")
        
        return
    
    def gameIsRunning(self, process_name: str='r5apex.exe') -> bool:    
        return process_name in [p.name() for p in process_iter()]
    
    def windowIsFocused(self, window_name: str='Apex Legends') -> bool:
        if self.gameIsRunning():
            return gw.getWindowsWithTitle(window_name)[0].isActive
        return False

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
KEYBINDS = {
    "m" : TrackerControls.RECORDING, 
    "pagedown" : TrackerControls.RECORDING, 
    "pageup" : TrackerControls.EXIT,
}
CONFIG = {
    # Companion features
    "trackDeaths": False,
    "autoQueue": False, # To be implemeneted

    # configuration
    "screenCaptureDelay": 0.5, # in seconds, delay between screen captures
    "keybinds": KEYBINDS,
}

if __name__ == "__main__":
    tracker = ApexTracker(CONFIG)

    if tracker.gameIsRunning():
        while True:
            event = keyboard.read_event()

            if event.event_type == keyboard.KEY_DOWN and event.name in CONFIG["keybinds"].keys():
                tracker.update(CONFIG["keybind"][event.name])
    else:
        log.error("Apex Legends is not running.")
 
    sys.exit(0)