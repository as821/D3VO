import numpy as np





class Point:
    # A point is a 3D location in global coordinate system, observed by multiple frames
    def __init__(self):
        self.frames = []        # set of keyframes where this point is visible
        self.id = -1            # TODO make me unique



class Frame:
    def __init__(self, image, depth, uncertainty, pose, brightness_params):
        # DepthNet and PoseNet outputs
        self.depth_map = depth
        self.uncertainty = uncertainty
        self.brightness_params = brightness_params
        self.pose = pose

        # Additional frame metadata
        self.image = image
        self.pts = []           # set of points visible in this keyframe
        self.id = -1            # TODO make me unique



    def calc_optimizer_weight(self, pt, alpha=0.5):
        """Calculate the weight for the specific point in this frame"""
        a2 = alpha**2
        return a2 / (a2 + np.linalg.norm(self.uncertainty(pt))**2)



