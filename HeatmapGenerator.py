import logging as log
import colorama

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os

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

    def getCoordsFromImage(self, img: Image, smap: Image) -> list:
        w, h = img.shape[:-1]

        res = cv2.matchTemplate(smap, img, cv2.TM_CCOEFF_NORMED)
        threshold = .35
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):  # Switch columns and rows
            cv2.rectangle(smap, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        
        # orb = cv2.ORB_create()
        # kp = orb.detect(img, None)
        # kp, des = orb.compute(img, kp)
        # # draw only keypoints location,not size and orientation
        # img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
        cv2.imwrite('result.png', smap)

    def generateHeatmap(self, smap: str, colormap: str='viridis') -> None:
        # Use opencv's ORB as an alternative to SIFT or SURF
        if not smap in self.CONFIG["maps"]:
            raise ValueError(f"Map name '{smap}' is not valid.")

        curr_map = smap if smap != "WORLO'S EDGE" else "WORLD'S EDGE"
        map_dir = os.path.join(self.CONFIG["dirPath"],os.path.join("maps", curr_map+".png"))
        scrn_dir = os.path.join(self.CONFIG["dirPath"], os.path.join("captures", curr_map))

        if not os.path.exists(map_dir):
            raise FileNotFoundError(f"Map image at: './maps/{curr_map}' does not exist.")
        if not os.path.exists(scrn_dir):
            raise FileNotFoundError(f"Directory './captures/{curr_map}' does not exist.")    
        
        map_img = cv2.imread(map_dir)
        screenshots = [img for img in os.listdir(scrn_dir) if img.endswith('.png')]
        
        log.info(f"Generating heatmap data from './captures/{curr_map}/' ...")
        for img in screenshots:
            img_path = os.path.join(scrn_dir, img)
            img = cv2.imread(img_path)
            coords = self.getCoordsFromImage(img, map_img)
            while True:
                pass

        log.info(f"Done.")
        return
        