from frontend import Point, Frame
import numpy as np
import g2o

from frontend import match_frame_kps



class Map:
	"""Class to store and optimize over all current frames, points, etc."""
	def __init__(self, alpha=0.5, num_kf=7):
		self.frames = []
		self.points = []
		self.keyframes = []
		self.frame_idx = self.pt_idx = 0
		self.num_kf = num_kf
		
		# Optimization hyperparameter for weighting uncertainty of a pixel (D3VO Eq. 13)
		self.alpha = alpha


	def add_frame(self, frame):
		"""Add a Frame to the Map"""
		assert (type(frame) == Frame)
		ret = self.frame_idx
		self.frame_idx += 1
		self.frames.append(frame)
		return ret

	def add_point(self, pt):
		"""Add a Point to the Map"""
		assert (type(pt) == Point)
		ret = self.pt_idx
		self.pt_idx += 1
		self.points.append(pt)
		return ret

	def check_add_key_frame(self, frame, intrinsics):
		"""Check if the given Frame is a keyframe, if so add to list and evaluate marginalization."""
		if frame.id == 0:
			key_frame = True
		else:
			key_frame = self.check_key_frame(frame, intrinsics)

		if key_frame:
			self.keyframes.append(frame)
			self.marginalize()

		return key_frame

	def marginalize(self):
		"""Check if any of the keyframes are ready for marginalization"""
		latest_key_frame = self.keyframes[-1]
		max_dist = 0
		max_dist_idx = 0
		marginalized_count = 0

		# Can marginalize everything apart from the last two keyframes:
		for i in range(len(self.keyframes) - 1):
			l1, l2 = match_frame_kps(latest_key_frame, self.keyframes[i])
			if len(l2) / len(self.keyframes[i].kps) < 0.1:
				self.keyframes[i].marginalize = True
				marginalized_count += 1
			frame_dist = np.linalg.norm(latest_key_frame.image - self.keyframes[i].image)
			if frame_dist > max_dist:
				max_dist = frame_dist
				max_dist_idx = i

		if len(self.keyframes) > self.num_kf and marginalized_count == 0:
			self.keyframes[max_dist_idx].marginalize = True

	def check_key_frame(self, frame, intrinsics):
		last_key_frame = self.frames[-1]
		w_a = 0.0
		w_f = 0.6
		w_ft = 0.4
		assert(w_a + w_f + w_ft == 1)
		l1, l2 = match_frame_kps(last_key_frame, frame)

		# Compute homography to wrap points just for translation
		# Camera projection is given by x = K[R | Rt]X where
		# K[R | Rt] is the pose for the camera.
		# We find the rotation for camera by multiplying with inverse
		# of the intrinsic and taking just the first 3 rows and first 3
		# columns.
		# The homography is then given by R1 @ inv(R2) @ inv(K)
		R1 = (np.linalg.inv(intrinsics) @ last_key_frame.pose)[:3, :3]
		R2 = (np.linalg.inv(intrinsics) @ frame.pose)[:3, :3]
		homography_t = intrinsics[:3,:3] @ R1 @ np.linalg.inv(R2) @ np.linalg.inv(intrinsics[:3,:3])

		f = 0
		ft = 0
		a = 0

		for idx1, idx2 in zip(l1, l2):
			x1, y1 = last_key_frame.kps[idx1]
			x2, y2 = frame.kps[idx2]
			f += (x1 - x2) ** 2 + (y1 - y2) ** 2
			pt = homography_t @ np.array([x2, y2, 1]).reshape(3, 1)
			x_pt = pt[0] / pt[-1]
			y_pt = pt[1] / pt[-1]

			ft += (x1 - x_pt) ** 2 + (y1 - y_pt) ** 2

		f /= len(l1)
		f = np.sqrt(f)
		ft /= len(l1)
		ft = np.sqrt(ft)

		return (w_f * f + w_ft * ft + w_a * a) > 1


	def optimize(self, intrinsic, iter=6, verbose=False):
		"""Run hypergraph-based optimization over current Points and Frames. Work in progress..."""
		# create optimizer
		opt = g2o.SparseOptimizer()
		solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
		solver = g2o.OptimizationAlgorithmLevenberg(solver)
		opt.set_algorithm(solver)
		opt.set_verbose(verbose)

		opt_frames, opt_pts = {}, {}
	
		# add camera
		f = intrinsic[0, 0]
		cx = intrinsic[0, 2]
		cy = intrinsic[1, 2]       
		assert intrinsic[0, 0] == intrinsic[1, 1]		# fx == fy
		cam = g2o.CameraParameters(f, (cx, cy), 0)         
		cam.set_id(0)
		opt.add_parameter(cam)  

		if len(self.keyframes) >= 7:
			self.keyframes[0].marginalize = True

		# set up frames as vertices
		for idx, f in enumerate(self.keyframes):
			# add frame to the optimization graph as an SE(3) pose
			v_se3 = g2o.VertexD3VOFramePose(f.image)
			v_se3.set_estimate(g2o.SE3Quat(f.pose[0:3, 0:3], f.pose[0:3, 3])) 	# .Isometry3d() use frame pose estimate as initialization
			v_se3.set_id(f.id * 2)			# even IDs only

			if idx == 0:
				v_se3.set_fixed(True)       # Hold first frame constant

			if f.marginalize:
				v_se3.set_marginalized(True)

			opt.add_vertex(v_se3)
			opt_frames[f] = v_se3

		# set up point edges between frames and depths
		kpts = self.keypoints()
		for p in kpts:
			host_frame, host_uv_coord = kpts[p][0][0], kpts[p][0][0].optimizer_kps[kpts[p][0][1]]
			pt = g2o.VertexD3VOPointDepth(host_uv_coord[0], host_uv_coord[1])
			pt.set_id(p.id * 2 + 1)		# odd IDs, no collisions with frame ID

			# unproject point with depth estimate onto 3D world using the host frame depth estimate
			host_depth_est = host_frame.depth[host_uv_coord[0]][host_uv_coord[1]]
			pt.set_estimate(host_depth_est)			
			
			pt.set_fixed(False)
			opt_pts[p] = pt
			opt.add_vertex(pt)

			# host frame connects to every edge involving this point
			for f in kpts[p][1:]:
				edge = g2o.EdgeProjectD3VO() 								
				edge.resize(3)
				edge.set_vertex(0, pt)										# connect to depth estimate
				edge.set_vertex(1, opt_frames[host_frame])					# connect to host frame
				edge.set_vertex(2, opt_frames[f[0]])						# connect to frame where point was observed
				
				# Incorporate uncertainty into optimization (D3VO Eq.13)
				weight_mx = np.eye(3) * (self.alpha**2) / (self.alpha**2 + np.sqrt(host_frame.uncertainty[host_uv_coord[0]][host_uv_coord[1]])**2)

				edge.set_information(weight_mx)								# simplified setting, no weights so use identity
				edge.set_robust_kernel(g2o.RobustKernelHuber())
				edge.set_parameter_id(0, 0)
				opt.add_edge(edge)

		# run optimizer
		opt.initialize_optimization()
		opt.optimize(iter)

		# store optimization results 
		for p in kpts:
			# optimization gives unprojected point in 3D
			est = opt_pts[p].estimate()
			assert est >= 0
			p.update_host_depth(est)
			# print(est)
	
		for f in self.keyframes:
			est = opt_frames[f].estimate()
			f.pose = np.eye(4)
			f.pose[:3, :3] = est.rotation().matrix()
			f.pose[:3, 3] = est.translation()
			#print(f.pose)

		# TODO(as) need to adjust this! if a keyframe is marginalized, must remove all points inside of it from any other keyframes
		# TODO(as) also need to handle marginalization of keyframes in the middle of the window, not just at the end
		if self.keyframes[0].marginalize:
			self.keyframes = self.keyframes[1:]
		

	def keypoints(self):
		"""Return a list of the points that originate in a keyframe and connect to other keyframes"""
		# Pretend that all points in the oldest keyframe originate in that keyframe
		candidate = list(self.keyframes[0].pts.values())

		# Find set of all points that originate a keyframe (ignoring the last keyframe)
		for f in self.keyframes[1:-1]:
			for pt in f.pts.values():
				if pt.frames[0] == f:
					# If this frame is the point's host frame, make it a candidate
					candidate.append(pt)

		# Refine candidates, check that they connect to at least one of the other keyframes
		kf = set(self.keyframes)
		keypoints = {}
		for p in candidate:
			local = []
			for idx, f in enumerate(p.frames):
				if f in kf:
					# Store frame as well as its index in the Point's frame list
					local.append((f, idx))

			# Only use a point if it connects to more than one keypoint
			if len(local) > 1:
				keypoints[p] = local

		return keypoints

