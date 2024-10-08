import logging as log

import argparse
import time
import sys
import os

import keyboard
import colorama
import pytesseract
from pytesseract import TesseractNotFoundError

from ApexTracker import ApexTracker, GameState, TrackerControls
from HeatmapGenerator import HeatmapGenerator

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

    # configuration
    "maps": APEX_MAPS,
    "screenCaptureDelay": .5, # in seconds, delay between screen captures
    "keybinds": KEYBINDS,
    "dirPath": DIR_PATH,
    "dirDeathCapture": CAPTURE_PATH,

    "videoAnalisys": False,
    "videoPath": None,

    # debug settings
    "debug": False,
    "debug_ignore_focus": False,
}
ARGUMENTS = [
    {
        "options": ["-d","--debug"],
        "kwargs": {
            "help": "Run in debug mode.",
            "action": "store_true"
        }
    },
    {
        "options": ["-t","--tracker"],
        "kwargs": {
            "help": "Run the ApexTracker",
            "action": "store_true"
        }
    },
    {
        "options": ["-g","--generate"],
        "kwargs": {
            "help": "Run the Heatmap generator",
            "action": "store_true"
        }
    },
    {
        "options": ["-s","--source"],
        "kwargs": {
            "help": "Analyze provided video file",
            "action": "store_true"
        }
    }
]

def argumentHelpFormater(prog) -> str:
    """Custom help formatter for argparse."""
    return argparse.HelpFormatter(prog, max_help_position=46, width=100)

def parseArguments() -> argparse.Namespace:
    """Parse the command line arguments and return the namespace object."""

    parser = argparse.ArgumentParser(
        formatter_class=argumentHelpFormater,
        description='Select which features to run.')
    
    for arg in ARGUMENTS:
        parser.add_argument(*arg["options"], **arg["kwargs"])

    return parser.parse_args()

def devSaveDeathsFromScreenshotDir() -> None:
    from PIL import Image
    tracker = ApexTracker(CONFIG)

    screen_dir = input("Enter the directory of the screenshots: ")
    if not os.path.exists(screen_dir):
        log.error("Directory '%s' does not exist.", screen_dir)
        sys.exit(1)

    curr_map = None
    while curr_map not in APEX_MAPS:
        curr_map = input("Enter map name: ")
        if curr_map not in APEX_MAPS:
            log.error("Map name '%s' is not valid.", curr_map)
            sys.exit(1)

    images = [fname for fname in os.listdir(screen_dir) if fname.endswith('.png')]
    try:
        for img in images:
            tracker.devGenerateDeathFromScreen(curr_map, Image.open(os.path.join(screen_dir, img)))
    except Exception as e:
        log.error("An error occured when saving death locations: %s", exc_info=e)
    finally:
        log.info("Saved all death locations.")

def startApexTracker() -> None:
    """Start the Apex Tracker process."""
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
                    tracker.recording_delay = CONFIG["screenCaptureDelay"]
                    tracker.last_capture = None
                    tracker.recording = True

                elif new_state in [GameState.KNOCKED, GameState.DEAD] and CONFIG["trackDeaths"]:
                    if tracker.last_capture:
                        tracker.saveDeathLocation(tracker.last_capture)
                        tracker.last_capture = None

                elif new_state == GameState.IN_QUEUE:
                    # GameState.LOBBY could be skipped if the user instantly queues up
                    # so then the map probably stays the same
                    if tracker.last_capture:
                        tracker.updateMap(tracker.last_capture.crop((50, 859, 326, 896)))

                log.info("Changing game state to %s%s%s", colorama.Fore.GREEN, new_state.name, colorama.Style.RESET_ALL)
                tracker.STATE = new_state

            if tracker.STATE == GameState.LOBBY:
                tracker.recording_delay = .05 # lower delay for lobby
                tracker.last_capture = tracker.captureScreen()

            time.sleep(tracker.recording_delay)
    else:
        log.error("Apex Legends is not running.")

    return

def startHeatmapGenerator() -> None:
    """Generate a heatmap from the saved death locations of a given map."""
    gen = HeatmapGenerator(CONFIG)

    if CONFIG["debug"]:
        log.info("Debug flag detected! Select option to continue:")
        log.info("1. Generate heatmap")
        log.info("2. Dev test object recognition")
        choice = input("Enter choice: ")

        try:
            choice = int(choice)

            match choice:
                case 1:
                    curr_map = gen.selectMap()
                    gen.generate(curr_map)
                case 2:
                    gen.devTestObjectRecognition()
        except ValueError:
            log.error("Invalid input. Exiting...")
            sys.exit(1)
    else:
        curr_map = gen.selectMap()
        try:
            gen.generate(curr_map)
        except FileNotFoundError as e:
            log.error("An error occured: %s%s%s", colorama.Fore.RED, e, colorama.Style.RESET_ALL)

    return

def startVideoAnalysis() -> None:
    """Start the video analysis process."""
    tracker = ApexTracker(CONFIG)

    while tracker.is_running:
        new_state = tracker.checkGameState()
        if tracker.STATE != new_state and new_state is not None:
            if new_state in [GameState.IN_DROPSHIP, GameState.ALIVE]:
                tracker.video_frames_to_skip = int(tracker.video_fps* CONFIG["screenCaptureDelay"])
                tracker.last_capture = None
                tracker.recording = True

            elif new_state in [GameState.KNOCKED, GameState.DEAD] and CONFIG["trackDeaths"]:
                if tracker.last_capture:
                    tracker.saveDeathLocation(tracker.last_capture)
                    tracker.last_capture = None

            elif new_state == GameState.IN_QUEUE:
                # GameState.LOBBY could be skipped if the user instantly queues up
                # so then the map probably stays the same
                if tracker.last_capture:
                    tracker.updateMap(tracker.last_capture.crop((50, 859, 326, 896)))

            log.info("Changing game state to %s%s%s", colorama.Fore.GREEN, new_state.name, colorama.Style.RESET_ALL)
            tracker.STATE = new_state
        if tracker.STATE == GameState.LOBBY:
            tracker.video_frames_to_skip = int(tracker.video_fps * .05) # lower delay for lobby
            tracker.last_capture = tracker.captureScreen()

    log.info("Video analysis complete.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select which features to run.')
    parser.add_argument('--debug', help='Run in debug mode.', nargs='*')
    parser.add_argument('-t', '--tracker', help="Run the ApexTracker", nargs='*')
    parser.add_argument('-g', '--generate', help="Run the Heatmap generator", nargs='*')
    parser.add_argument('-s', '--source', help="Analyze provided video file", nargs='*')
    args = parser.parse_args()

    # check if 'debug' arg was supplied or is it set to 'True'
    if args.debug:
        CONFIG["debug"] = True

    # return an error if both 'tracker' and 'generate' args are supplied
    if sum(arg is not None for arg in [args.tracker, args.generate, args.source]) > 1:
        print("Please select only one feature to run.")
        sys.exit(1)

    log.basicConfig(
        level=log.DEBUG if CONFIG["debug"] else log.INFO,
        #filename='apex_tracker.log', # uncomment to disable console logs
        format='[%(levelname)s] %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        )
    log.debug('Debug mode enabled.')

    # check if pytesseact is installed correctly
    try:
        pytesseract.pytesseract.tesseract_cmd = os.path.join(DIR_PATH, os.path.join('tesseract', 'tesseract.exe'))
        pytesseract.get_tesseract_version()
        log.debug("Tesseract detected, version: %s", pytesseract.get_tesseract_version())
    except TesseractNotFoundError as e:
        log.error("Tesseract installation not found. Make sure it's installed and in your PATH.")
        sys.exit(1)
    except Exception as e:
        log.error("An error occured when loading tesseract: \n", exc_info=e)
        sys.exit(1)

    if args.tracker:
        startApexTracker()
    elif args.generate:
        startHeatmapGenerator()
    elif args.source is not None:
        CONFIG['videoAnalisys'] = True
        CONFIG['videoPath'] = args.source[0]
        startVideoAnalysis()
    else: # by default start the Apex Tracker
        startApexTracker()

    sys.exit(0)
