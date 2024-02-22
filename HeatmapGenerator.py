import logging as log
import colorama

from PIL import Image
import numpy as np
import cv2

class HeatmapGenerator:
    def __init__(self, config) -> None:
        self.CONFIG = config

    def selectMap(self) -> str:
        while True:
            try:
                map_list = ",\n".join([f"{i+1}. {map_name}" for i, map_name in enumerate(self.CONFIG['maps'])])
                log.info(f"Maps:\n{map_list}")
                selected_map = input("Select a map to generate a heatmap for (leave empty for all maps): ")

                selected_map = int(selected_map)
                if not selected_map or selected_map < 1 or selected_map > len(map_list):
                    raise ValueError

                return self.CONFIG['maps'][selected_map-1]
            
            except ValueError:
                log.error("Invalid input. Please select a valid option!")

    def generateHeatmap(self, curr_map: str, colormap: str='viridis') -> None:
        if not curr_map in self.CONFIG["maps"]:
            log.error(f"Map '{map}' does not exist.")
            return

        # Use opencv's ORB as an alternative to SIFT or SURF

        log.info(f"Generating Heatmap for map {curr_map}")
        