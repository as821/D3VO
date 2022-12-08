from frontend import Point, Frame
import numpy as np
import g2o

from frontend import match_frame_kps


class Map:
	"""Maintains and optimizes over all frames, points, and keyframes."""
	def __init__(self, alpha=0.5, num_kf=7, trajectory_scale=30):
		self.frames = []
		self.points = []
		self.keyframes = []
		self.frame_idx = self.pt_idx = 0

		# Maximum number of keyframes, manages size of optimization window
		self.num_kf = num_kf

		# Hyperparameter for translation scaling
		self.trajectory_scale = trajectory_scale
		
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


	def check_add_key_frame(self, frame):
		"""Check if the given Frame should be a keyframe, if so add it to the keyframe list and evaluate marginalization."""
		if frame.id == 0:
			# Always make the first frame a keyframe
			key_frame = True
		else:
			key_frame = self.check_key_frame(frame)

		if key_frame:
			self.keyframes.append(frame)
			
		# Simple marginalization policy to maintain optimization window size
		if len(self.keyframes) >= self.num_kf:
			self.keyframes[0].marginalize = True

		return key_frame


	def check_key_frame(self, frame):
		"""Return True if the given Frame should be made a keyframe."""
		last_key_frame = self.keyframes[-1]
		w_a = 0.0
		w_f = 0.6
		w_ft = 0.4
		assert(w_a + w_f + w_ft == 1)
		l1, l2 = match_frame_kps(last_key_frame, frame)

		# Compute homography to wrap points just for translation
		global_poses = self.relative_to_global()
		if last_key_frame.id == 0:
			# Relative to global does not include the identity pose for the first frame, indices are off by 1 compared to IDs
			R1 = np.eye(3)
		else:
			R1 = global_poses[last_key_frame.id - 1][:3, :3]
		R2 = global_poses[frame.id - 1][:3, :3]
		homography_t =  R1 @ np.linalg.inv(R2)

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
		"""Run hypergraph-based optimization over current Points and Frames. Uses custom 
		VertexD3VOFramePose, VertexD3VOPointDepth, and EdgeProjectD3VO g2o types implemented in 
		C++ and used here through pybind to implement the backend loss described by D3VO paper."""
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

		# set up frames as vertices
		for idx, f in enumerate(self.keyframes):
			# add frame to the optimization graph as an SE(3) pose
			transform = (f.a * f.image + f.b).squeeze()
			v_se3 = g2o.VertexD3VOFramePose(transform)
			v_se3.set_estimate(g2o.SE3Quat(f.pose[0:3, 0:3], f.pose[0:3, 3]))
			v_se3.set_id(f.id * 2)			# even IDs only (avoid ID conflict with points)

			if idx == 0:
				# Hold first frame in the window constant
				v_se3.set_fixed(True)       

			if f.marginalize:
				# optimization library crashes if we marginalize a frame that is not at the beginning of the window (oldest keyframe)
				assert idx == 0				
				v_se3.set_marginalized(True)

			opt.add_vertex(v_se3)
			opt_frames[f] = v_se3

		# create a dictionary of keypoints that connect keyframes and add edges to the optimizer between this 
		# point, its host keyframe, and another keyframe
		kpts = self.keypoints()
		for p in kpts:
			# initialize point estimate with the depth estimate of its host frame
			host_frame, host_uv_coord = kpts[p][0][0], kpts[p][0][0].optimizer_kps[kpts[p][0][1]]
			pt = g2o.VertexD3VOPointDepth(host_uv_coord[0], host_uv_coord[1])
			pt.set_id(p.id * 2 + 1)		# odd IDs, no collisions with frame ID
			pt.set_estimate(host_frame.depth[host_uv_coord[0]][host_uv_coord[1]])			
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
				edge.set_information(weight_mx)					
				edge.set_robust_kernel(g2o.RobustKernelHuber())
				edge.set_parameter_id(0, 0)
				opt.add_edge(edge)

		# run optimizer if there are keypoints
		if len(kpts) > 0:
			opt.initialize_optimization()
			opt.optimize(iter)

			# store optimization results in our objects
			for p in kpts:
				est = opt_pts[p].estimate()
				assert est >= 0
				p.update_host_depth(est)
		
			for idx, f in enumerate(self.keyframes):
				est = opt_frames[f].estimate()
				f.pose = np.eye(4)
				f.pose[:3, :3] = est.rotation().matrix()
				f.pose[:3, 3] = est.translation()

			# Library only supports marginalizing keyframe at the beginning of the window (oldest keyframe)
			if self.keyframes[0].marginalize:
				self.keyframes = self.keyframes[1:]
				# Mark all points in a marginalized frame as invalid
				for pt in f.pts.values():
					pt.valid = False

	def keypoints(self):
		"""Return a dictionary of the Points that originate in a keyframe and connect to other keyframes."""
		# Pretend that all valid points in the oldest keyframe originate in that keyframe
		candidate = [p for p in list(self.keyframes[0].pts.values()) if p.valid]

		# Find set of all points that originate a keyframe (ignoring the last keyframe)
		for f in self.keyframes[1:-1]:
			for pt in f.pts.values():
				if pt.frames[0] == f and pt.valid:
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

			# Only use a point if it connects to more than one keyframe
			if len(local) > 1:
				keypoints[p] = local
		return keypoints

	def relative_to_global(self):
		"""Convert the relative pose stored in frames into a global pose."""
		pred_pose = []
		for idx, f in enumerate(self.frames[1:]):		
			if idx > 1:
				pred_pose.append(np.dot(pred_pose[idx-1], np.linalg.inv(self.frames[idx].pose)))
			else:
				pred_pose.append(np.linalg.inv(f.pose))

		for t in range(len(pred_pose)):
			pred_pose[t][:3, 3] *= self.trajectory_scale
		return pred_pose

