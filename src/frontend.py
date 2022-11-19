import numpy as np
import cv2

NUM_FEATURE = 250
FEATURE_QUALITY = 0.1


def extract_features(img):
    # detection
    orb = cv2.ORB_create()
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), NUM_FEATURE, qualityLevel=FEATURE_QUALITY, minDistance=7)

    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
    kps, des = orb.compute(img, kps)

    # return pts and des
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des



def match_frame_kps(f1, f2):
    """Match keypoints in the given frames"""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    # Lowe's ratio test (and remove duplicates)
    idx1, idx2 = [], []
    s1, s2 = set(), set()
    for m,n in matches:
        if m.distance < 0.75 * n.distance and m.distance < 32 and m.queryIdx not in s1 and m.trainIdx not in s2:
            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)
            s1.add(m.queryIdx)
            s2.add(m.trainIdx)
    assert len(set(idx1)) == len(idx1)
    assert len(set(idx2)) == len(idx2)
    return np.array(idx1), np.array(idx2)





class Point:
    # A point is a 3D location in global coordinate system, observed by multiple frames
    def __init__(self, map):
        self.frames = []        # set of keyframes where this point is visible
        self.idxs = []          # index for the kps/des lists of the corresponding frame. Parallel list to self.frames
        self.id = map.add_point(self)

    def add_observation(self, frame, idx):
        """Add a Frame where this Point was observed"""
        assert idx not in frame.pts
        assert frame not in self.frames
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)




class Frame:
    def __init__(self, map, image, depth, uncertainty, pose, brightness_params):
        self.depth_map = depth
        self.uncertainty = uncertainty
        self.brightness_params = brightness_params
        self.pose = pose

        self.image = image
        self.id = map.add_frame(self)       # get an ID from the map

        # Run frontend keypoint extractor
        self.kps, self.des = extract_features(image)
        self.pts = {}                       # map kps/des list index to corresponding Point object   



