import numpy as np
from frontend import extract_features




class Point:
    # A point is a 3D location in global coordinate system, observed by multiple frames
    def __init__(self, pid):
        self.frames = []        # set of keyframes where this point is visible
        self.id = pid



class Frame:
    def __init__(self, map, image, depth, uncertainty, pose, brightness_params):
        self.depth_map = depth
        self.uncertainty = uncertainty
        self.brightness_params = brightness_params
        self.pose = pose

        self.image = image
        # self.pts = []                   
        self.id = map.add_frame(self)       # get an ID from the map

        # Run frontend keypoint extractor
        self.kpus, self.des = extract_features(image)
        self.pts = [None]*len(self.kpus)    # set of points visible in this keyframe


    def calc_optimizer_weight(self, pt, alpha=0.5):
        """Calculate the weight for the specific point in this frame"""
        a2 = alpha**2
        return a2 / (a2 + np.linalg.norm(self.uncertainty(pt))**2)



