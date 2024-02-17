from ApexTracker import ApexTracker, GameState, TrackerControls

import pytesseract
import keyboard
import time
import sys
import os

import logging as log

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

def startApexTracker():
    pytesseract.pytesseract.tesseract_cmd = os.path.join(DIR_PATH, os.path.join('tesseract', 'tesseract.exe'))
    tracker = ApexTracker(CONFIG, ignore_checks=True)

    for key in tracker.CONFIG["keybinds"].keys():
        if tracker.CONFIG["keybinds"][key] == TrackerControls.RECORDING:
            keyboard.add_hotkey(key, lambda: tracker.pauseRecording())
        else:
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

                log.info(f"Changing game state to '{new_state.name}'")
                tracker.STATE = new_state
            
            if tracker.STATE == GameState.LOBBY:
                tracker.last_capture = tracker.captureScreen()
                
            time.sleep(CONFIG["screenCaptureDelay"])
    else:
        log.error("Apex Legends is not running.")
 
    sys.exit(0)

if __name__ == "__main__":
    startApexTracker()