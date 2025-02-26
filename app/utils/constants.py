from enum import Enum

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