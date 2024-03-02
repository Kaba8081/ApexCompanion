from ApexTracker import ApexTracker, GameState, TrackerControls
from HeatmapGenerator import HeatmapGenerator

import pytesseract
import argparse
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

    # debug settings
    "debug": False,
    "debug_ignore_focus": False,
}

def devSaveDeathsFromScreenshotDir() -> None:
    from PIL.Image import open
    tracker = ApexTracker(CONFIG)

    screen_dir = input("Enter the directory of the screenshots: ")
    if not os.path.exists(screen_dir):
        log.error(f"Directory '{screen_dir}' does not exist.")
        sys.exit(1)
    
    curr_map = None
    while curr_map not in APEX_MAPS:
        curr_map = input("Enter map name: ")
        if curr_map not in APEX_MAPS:
            log.error(f"Map name '{curr_map}' is not valid.")
            sys.exit(1)

    images = [fname for fname in os.listdir(screen_dir) if fname.endswith('.png')]
    try:
        for img in images:
            tracker.devGenerateDeathFromScreen(curr_map, open(os.path.join(screen_dir, img)))
    except Exception as e:
        log.error(f"An error occured when saving death locations: {e}")
        return
    finally:
        log.info("Saved all death locations.")
        return

def startApexTracker() -> None:
    pytesseract.pytesseract.tesseract_cmd = os.path.join(DIR_PATH, os.path.join('tesseract', 'tesseract.exe'))
    tracker = ApexTracker(CONFIG)

    for key in tracker.CONFIG["keybinds"].keys():
        if tracker.CONFIG["keybinds"][key] == TrackerControls.RECORDING:
            keyboard.add_hotkey(key, lambda: tracker.pauseRecording())
        else:
            keyboard.add_hotkey(key, tracker.update, args=(tracker.CONFIG["keybinds"][key],))

    if tracker.gameIsRunning() or CONFIG["debug"]:
        # while True:
        #     tracker.devFindOnScreen()
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

def startHeatmapGenerator() -> None:
    gen = HeatmapGenerator(CONFIG)

    curr_map = gen.selectMap()
    gen.generateHeatmap(curr_map)

    log.info(curr_map)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Select which features to run.')
    parser.add_argument('--debug', help='Run in debug mode.', nargs='*')
    parser.add_argument('-t', '--tracker', help="Run the ApexTracker", nargs='*')
    parser.add_argument('-g', '--generate', help="Run the Heatmap generator", nargs='*')
    args = parser.parse_args()

    # check if 'debug' arg was supplied or is it set to 'True'
    if args.debug is not None or args.debug is True:
        CONFIG["debug"] = True

    # return an error if both 'tracker' and 'generate' args are supplied
    if args.tracker is not None and args.generate is not None:
        log.error("Please select only one feature to run.")
        sys.exit(1)

    log.basicConfig(
        level=log.DEBUG if CONFIG["debug"] else log.INFO,
        #filename='apex_tracker.log', # uncomment to disable console logs
        format='[%(levelname)s] %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    log.debug(f'{args.debug} {CONFIG["debug"]}')

    if args.tracker is not None:
        startApexTracker()
    elif args.generate is not None:
        startHeatmapGenerator()
    else: # by default start the Apex Tracker
        #startApexTracker()
        #devSaveDeathsFromScreenshotDir()

        hg = HeatmapGenerator(CONFIG)
        hg.devTestObjectRecognition()
        