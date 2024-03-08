import logging as log
import colorama

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import errno
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

    def getCoordsFromImage(self, img: Image.Image, smap: Image.Image, ratio:float=.75) -> tuple:
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        detector = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=.04, edgeThreshold=10, sigma=1.6)
        
        kp1, des1 = detector.detectAndCompute(img, None)
        kp2, des2 = detector.detectAndCompute(smap, None)

        matches = matcher.knnMatch(des1, des2, k=2)

        good_matches_knn = []
        good_matches = []
        
        # Filter matches using the Lowe's ratio test
        for m,n in matches:
            if m.distance < ratio * n.distance:
                good_matches_knn.append([m])
                good_matches.append(m)

        result = cv2.drawMatchesKnn(img, kp1, smap, kp2, good_matches_knn, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        if len(good_matches) < 10:
            if self.CONFIG['debug']:
                plt.imshow(result, 'gray'),plt.show()
            raise ValueError("Not enough good matches found.")
        
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        center = np.mean(dst, axis=0)[0]
        center = tuple(map(int, center))

        # draw matches
        if self.CONFIG['debug']:
            # draw outline and the center of the object
            result = cv2.polylines(smap,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            result = cv2.circle(smap, center, 3, (166, 0, 255), -1)

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                            singlePointColor = None,
                            matchesMask = matchesMask, # draw only inliers
                            flags = 2)
            result = cv2.drawMatches(img,kp1,smap,kp2,good_matches,None,**draw_params)
            
            plt.imshow(result, 'gray'),plt.show()
        
        return center

    def GenerateHeatmap(self,
                        data:list, 
                        map_name: str, 
                        map_img: Image.Image, 
                        colormap: str = "viridis",
                        resolution: int = 100
                        ) -> None:
        BASE_RESOLUTION = 1015 # resolution of the map image
        scale = resolution / BASE_RESOLUTION

        heatmap_data = np.asarray([[0] * resolution] * resolution)
        for coord in data:
            x, y = coord
            heatmap_data[int(y*scale)][int(x*scale)] += 100

        total = np.sum(heatmap_data)
        heatmap_data = heatmap_data / total

        plt.clf()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(heatmap_data, interpolation="lanczos", zorder=1, cmap=colormap, alpha=.5, extent=(0, BASE_RESOLUTION, 0, BASE_RESOLUTION))    
        plt.imshow(np.flipud(map_img), zorder=0, cmap="gray",origin='lower')

        plt.savefig(
            f'{self.CONFIG["dirPath"]}/heatmap_{map_name}.png',
            dpi=300,
            pad_inches=0,
            transparent=True,
            bbox_inches='tight'
        )

        return
    
    def generate(self, smap: str, colormap: str='viridis') -> None:
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
        death_data = []

        log.info(f"Generating heatmap data from './captures/{curr_map}/' ...")
        for img_name in screenshots:
            img_path = os.path.join(scrn_dir, img_name)
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            coords = None

            try:
                coords = self.getCoordsFromImage(img, map_img)
            except ValueError as e:
                log.warning(f"Error for {img_name}: {e}")
            except Exception as e:
                log.error(f"An error occured for {img_name}: {e}")
                return
            finally:
                 if coords:
                    death_data.append(coords)
        if not len(death_data) > 0:
            log.error("No valid death data found.")
            return
        
        self.GenerateHeatmap(death_data, curr_map, map_img, colormap, 20)

        log.info(f"Done. Heatmap generated in './heatmap_{curr_map}.png'")
        return

    def devFindOnMap(
        self, 
        map_img: Image.Image,    # selected map image
        screen_img: Image.Image, # death screen image
        technique: str='ORB',    # object detection technique
        ratio: float=.75,            # Lowe's ratio test
        **kwargs: dict           # Arguments for the selected technique
     ) -> Image.Image:
        # This function returns an image with the object detection results

        detector = None
        matcher = None
        matches = None
        result = None

        # maybe usefull matchers:
        #matcher = cv2.BFMatcher()
        #matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

        match technique.upper():
            case 'ORB':
                #detector = cv2.ORB_create(**kwargs)
                detector = cv2.ORB_create(nfeatures=15, WTA_K=2, scaleFactor=2, patchSize=31)
            case 'SIFT':
                detector = cv2.SIFT_create(**kwargs)
            case 'KAAZE':
                pass
        
        # Find the keypoints and descriptors
        kp1, des1 = detector.detectAndCompute(screen_img, None)
        kp2, des2 = detector.detectAndCompute(map_img, None)

        match technique.upper():
            case 'ORB':
                matches = matcher.knnMatch(des1, des2, k=2)

                good_matches = []
                # Lowe's ratio test
                for m,n in matches:
                    if m.distance < ratio *n.distance:
                        good_matches.append([m])

                # Draw matches
                result = cv2.drawMatchesKnn(screen_img,  
                            kp1, map_img, kp2, good_matches, None, 
                            matchColor=(0, 255, 0), matchesMask=None, 
                            singlePointColor=(255, 0, 0), flags=0) 

            case 'SIFT':
                matches = matcher.match(des1, des2)

                # sort by distance
                matches = sorted(matches, key = lambda x:x.distance)

                # draw only 10 best matches
                img_matches = np.empty((max(screen_img.shape[0], map_img.shape[0]), screen_img.shape[1]+ map_img.shape[1], 3), dtype=np.uint8)
                result = cv2.drawMatches(screen_img,kp1,map_img,kp2,matches[:10],img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return result

    def devTestMapDisplayMenu(self, curr_conf: dict) -> None:
        log.debug("-- Object recognition test options --")
        for key in curr_conf.keys():
            log.debug(f"{colorama.Fore.GREEN}{key}{colorama.Style.RESET_ALL}: {curr_conf[key]}")
        log.debug("-----")
        
        return

    def devTestMapChangeSettings(self, 
                                 curr_conf: dict, 
                                 default_args: dict, 
                                 avail_methods: list=["SIFT", "ORB", "AKAZE"]
                                 ) -> tuple:
        new_conf = {
            'method': curr_conf['method'],
            'ratio': curr_conf['ratio'],
            'args': curr_conf['args'],
            'map': curr_conf['map'],
            'file': curr_conf['file']
        }

        while True:
            log.debug("-- Change settings --")
            for i, key in enumerate(new_conf.keys()):
                log.debug(f"{i+1}. Change {key}")
            log.debug("6. Exit")
            choice = input("Select an option: ")

            try:
                choice = int(choice)
                if choice < 1 or choice > 6:
                    raise ValueError
            except ValueError:
                log.error("Invalid option!")
                continue
            finally:
                match choice:
                    case 1: # Change method
                        log.debug(f"Available methods: {', '.join(avail_methods)}")
                        try: 
                            temp = input("New method:")
                            if temp.upper() not in avail_methods:
                                raise ValueError
                            new_conf["method"] = temp.upper()

                        except ValueError:
                            log.error("Invalid method!")
                            continue
                    case 2: # Change ratio
                        temp = input("New ratio:")
                        try:
                            temp = float(temp)
                            if temp < 0 or temp > 1:
                                raise ValueError
                            new_conf["ratio"] = temp

                        except ValueError:
                            log.error("Invalid ratio!")
                            continue    
                    case 3: # Change arguments
                        temp = input("Add custom arguments (format: key1=value1,key2=value2,...): ")
                        try:
                            temp = temp.split(",")
                            temp_args = {}
                            for argument in temp:
                                argument = argument.split("=")
                                
                                # try to set the arguments value to int,
                                # if an exception would be thrown set it as str
                                try:
                                    temp_args[argument[0]] = int(argument[1])
                                except:
                                    temp_args[argument[0]] = argument[1]

                            new_conf["args"] = temp_args
                        except IndexError:
                            log.error("Wrong format!")
                    case 4: # Change map
                        log.debug("-- Maps --")
                        log.debug(", ".join(self.CONFIG['maps']))
                        temp = input("New map:")
                        try:
                            if temp.upper() not in self.CONFIG['maps']:
                                raise ValueError
                            new_conf["map"] = temp.upper()
                        except ValueError:
                            log.error("Invalid map!")
                            continue
                    case 5: # Change path to file
                        temp = input("New path to file (relative to map's folder):")
                        new_path = os.path.join(self.CONFIG["dirPath"], 'captures')
                        new_path = os.path.join(new_path, curr_conf["map"])
                        new_path = os.path.join(new_path, temp)

                        try:
                            if not os.path.exists(temp):
                                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), new_path)
                            curr_conf['file'] = new_path
                        except Exception as e:
                            log.error(f"An exception occured while locating file: \n{e}")
                            continue
                    case 6: # Exit
                        break
                    
        return new_conf
    
    def devTestMapSelectImages(self) -> tuple:
        map_img, screen_img = None, None
        
        map_img = os.path.join(self.CONFIG['dirPath'],os.path.join("maps","WORLD'S EDGE.png"))
        map_img = cv2.imread(map_img, cv2.IMREAD_GRAYSCALE)
        screen_img = os.path.join(self.CONFIG['dirPath'],"dev_screen.png")
        screen_img = cv2.imread(screen_img, cv2.IMREAD_GRAYSCALE)

        # TODO: implement a way to select the images from the directory

        return map_img, screen_img

    def devTestObjectRecognition(self) -> None:
        avail_methods = ['SIFT', 'ORB', 'AKAZE']
        default_args = {
            'SIFT': {'nfeatures':0, 'nOctaveLayers':3, 'contrastThreshold':.04, 'edgeThreshold':10, 'sigma':1.6},
            'ORB': {'nfeatures':15, 'WTA_K':2, 'scaleFactor':2, 'patchSize':31},
            'AKAZE': {}
        }
        
        # default settings
        curr_conf = {
            'method': 'SIFT',
            'ratio': .75,
            'args': default_args['SIFT'],
            'map': "WORLD'S EDGE",
            'file': "dev_screen.png"
        }
        
        while True:
            self.devTestMapDisplayMenu(curr_conf)
            log.debug("1. Change settings")
            log.debug("2. Test with selected settings")
            log.debug("3. Exit")
            choice = input("Select an option: ")
            
            try:
                choice = int(choice)
                if choice < 1 or choice > 3:
                    raise ValueError
            except ValueError:
                log.error("Invalid option!")
                continue
            finally:
                match choice:
                    case 1: # Change settings
                        curr_conf = self.devTestMapChangeSettings(curr_conf, avail_methods, default_args)

                    case 2: # Test with selected settings
                        map_img, screen_img = self.devTestMapSelectImages()
                        result_img = self.devFindOnMap(map_img, screen_img, curr_conf["method"], curr_conf["ratio"], **curr_conf["args"])

                        plt.imshow(result_img), plt.show()
                    case 3: # Exit
                        break
    
        return
       