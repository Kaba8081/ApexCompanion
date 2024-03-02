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

    def getCoordsFromImage(self, img: Image, smap: Image) -> list: # TODO: Find the correct initialization settings
        # Match template solution
        # w, h = img.shape[:-1]

        # res = cv2.matchTemplate(smap, img, cv2.TM_CCOEFF_NORMED)
        # threshold = .35
        # loc = np.where(res >= threshold)
        # for pt in zip(*loc[::-1]):  # Switch columns and rows
        #     cv2.rectangle(smap, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        
        # openCV's ORB solution 1.0
        # orb = cv2.ORB_create()
        # kp = orb.detect(img, None)
        # kp, des = orb.compute(img, kp)
        # # draw only keypoints location,not size and orientation
        # img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
        
        # openCV's ORB solution 2.0
        # Initiate SIFT detector
        orb = cv2.ORB_create(15, WTA_K=2, scaleFactor=2, patchSize=31)
        #sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(img,None)
        kp2, des2 = orb.detectAndCompute(smap,None)
        # create BFMatcher object
        bf = cv2.BFMatcher()#cv2.NORM_HAMMING, crossCheck=True)
        #matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        # Match descriptors.
        #matches = bf.match(des1,des2)
        matches = bf.knnMatch(des1,des2, k=2)

        # Sort them in the order of their distance.
        #matches = sorted(matches, key = lambda x:x.distance) 
        good = []
        # matched_image = cv2.drawMatchesKnn(img,  
        #    kp1, smap, kp2, matches, None, 
        #    matchColor=(0, 255, 0), matchesMask=None, 
        #    singlePointColor=(255, 0, 0), flags=0) 
        
        for m,n in matches:
            if m.distance < 0.95*n.distance:
                good.append(m)

        # Draw first 10 matches.
        img_matches = np.empty((max(img.shape[0], smap.shape[0]), img.shape[1]+img.shape[1], 3), dtype=np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # smap = cv2.cvtColor(smap, cv2.COLOR_GRAY2RGB)
        img3 = cv2.drawMatches(img,kp1,smap,kp2,good[:10],img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.imshow(img3),plt.show()
        cv2.imwrite('result.png', img3)
        cv2.imwrite('template.png', img)

    def devFindOnMap(
        self, 
        screen_img: Image.Image, # death screen image
        map_img: Image.Image,    # selected map image
        technique: str='ORB',    # object detection technique
        ratio: float=.75,            # Lowe's ratio test
        **kwargs: dict           # Arguments for the selected technique
     ) -> Image.Image:
        # This function returns an image with the object detection results

        detector = None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = None
        good_matches = []
        result = None

        match technique.upper():
            case 'ORB':
                detector = cv2.orb.ORB_create(args=kwargs)
            case 'SIFT':
                detector = cv2.SIFT_create(args=kwargs)
            case 'KAAZE':
                pass
        
        # Find the keypoints and descriptors
        kp1, des1 = detector.detectAndCompute(screen_img, None)
        kp2, des2 = detector.detectAndCompute(map_img, None)

        match technique.upper():
            case 'ORB':
                matches = matcher.match(des1, des2)

                for m,n in matches:
                    if m.distance < ratio *n.distance:
                        good_matches.append([m])

            case 'SIFT':
                matches = matcher.knnMatch(des1, des2, k=2)

                for m,n in matches:
                    if m.distance < ratio *n.distance:
                        good_matches.append(m)

        img_matches = np.empty((max(screen_img.shape[0], map_img.shape[0]), screen_img.shape[1]+ map_img.shape[1], 3), dtype=np.uint8)
        # draw only 10 best matches
        result = cv2.drawMatches(screen_img,kp1,map_img,kp2,good_matches[:10],img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return result

    def devTestObjectRecognition(self) -> None:
        # TODO: implement supplying custom settings for the object recognition in order to test the different techniques
        pass

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
        
        map_img = cv2.imread(map_dir,cv2.IMREAD_GRAYSCALE)
        screenshots = [img for img in os.listdir(scrn_dir) if img.endswith('.png')]
        
        log.info(f"Generating heatmap data from './captures/{curr_map}/' ...")
        for img in screenshots:
            img_path = os.path.join(scrn_dir, img)
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            coords = self.getCoordsFromImage(img, map_img)

            # TODO: Generate heatmap data from the coordinates

        log.info(f"Done.")
        return
        