import numpy as np
import cv2

# Feature extraction hyperparameters
NUM_FEATURE = 3000
FEATURE_QUALITY = 0.01


def extract_features(img):
	"""Extract ORB features from given image, return keypoints and their descriptors."""
	# detection
	orb = cv2.ORB_create()
	pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), NUM_FEATURE, qualityLevel=FEATURE_QUALITY, minDistance=7)

	# extraction
	kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
	kps, des = orb.compute(img, kps)

	# return pts and des
	return np.array([(int(kp.pt[0]), int(kp.pt[1])) for kp in kps]), des


def match_frame_kps(f1, f2):
	"""Match ORB keypoints in the given frames"""
	bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	matches = bf.knnMatch(f1.des, f2.des, k=2)

	# Lowe's ratio test (and remove duplicates)
	idx1, idx2 = [], []
	pts1, pts2 = [], []
	s1, s2 = set(), set()
	for m,n in matches:
		if m.distance < 0.75 * n.distance and m.distance < 32 and m.queryIdx not in s1 and m.trainIdx not in s2:
			idx1.append(m.queryIdx)
			idx2.append(m.trainIdx)
			s1.add(m.queryIdx)
			s2.add(m.trainIdx)

			pts1.append(f1.kps[m.queryIdx])
			pts2.append(f2.kps[m.trainIdx])

	assert len(set(idx1)) == len(idx1)
	assert len(set(idx2)) == len(idx2)
	return idx1, idx2



class Point:
	# A point is a 3D location in global coordinate system, observed by multiple frames
	def __init__(self, map):
		self.frames = []        # set of keyframes where this point is visible
		self.idxs = []          # index for the kps/des lists of the corresponding frame. Parallel list to self.frames
		self.id = map.add_point(self)
		self.valid = True       # point becomes invalid when a Frame it appears in is marginalized
	
	def get_host_frame(self):
		# Host frame for this point is the first frame it is observed in
		return self.frames[0], self.frames[0].optimizer_kps[self.idxs[0]]

	def update_host_depth(self, depth):
		host_frame, host_uv_coord = self.get_host_frame()
		host_frame.depth[host_uv_coord[0]][host_uv_coord[1]] = depth

	def add_observation(self, frame, idx):
		"""Add a Frame where this Point was observed"""
		assert idx not in frame.pts
		assert frame not in self.frames
		assert idx < len(frame.optimizer_kps)

		frame.pts[idx] = self
		self.frames.append(frame)
		self.idxs.append(idx)
		

class Frame:
	def __init__(self, map, image, depth, uncertainty, pose, brightness_params):
		self.id = map.add_frame(self)       # get an ID from the map
		self.image = image

		self.depth = depth
		self.uncertainty = uncertainty
		self.a = brightness_params[0]
		self.b = brightness_params[1]
		self.pose = pose

		self.marginalize = False

		# Run frontend keypoint extractor
		self.kps, self.des = extract_features(image)
		self.pts = {}                       # map kps/des list index to corresponding Point object   

		# Optimizer expects keypoints in a different coordinate ordering
		self.optimizer_kps = [(k[1], k[0]) for k in self.kps]

		# Ensure that u/v coordinates of keypoints match the image/depth/uncertainty dimension shape (catch these issues at the source!)
		assert all([p[0] >= 0 and p[0] <= self.image.shape[0] and p[1] >= 0 and p[1] <= self.image.shape[1] for p in self.optimizer_kps])