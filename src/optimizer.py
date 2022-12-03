from frontend import Point, Frame
from helper import project, unproject

import numpy as np
import g2o



class Map:
	"""Class to store and optimize over all current frames, points, etc."""
	def __init__(self, alpha=0.5):
		self.frames = []
		self.points = []

		self.keyframes = []

		self.frame_idx = self.pt_idx = 0
		
		# Optimization hyperparameter for weighting uncertainty of a pixel (D3VO Eq. 13)
		self.alpha = alpha

	def add_frame(self, frame):
		"""Add a Frame to the Map"""
		# TODO assumes no frame removal in ID assignment
		assert (type(frame) == Frame)
		ret = self.frame_idx
		self.frame_idx += 1
		self.frames.append(frame)
		return ret

	def add_point(self, pt):
		"""Add a Point to the Map"""
		# TODO assumes no point removal in ID assignment
		assert (type(pt) == Point)
		ret = self.pt_idx
		self.pt_idx += 1
		self.points.append(pt)
		return ret

	def optimize(self, intrinsic, iter=6, verbose=True):
		"""Run hypergraph-based optimization over current Points and Frames. Work in progress..."""
		# create optimizer (TODO just following example, likely incorrect for D3VO)
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

		# Update poses of all frames between keyframes + after the last keyframe
		self.recompute_global_poses()

		# Remove marginalized keyframe
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


	def recompute_global_poses(self):
		"""After a bundle adjustment, recompute relative poses of all frames that come after the first keyframe."""
		for kf_idx in range(len(self.keyframes)):
			# Adjust relative poses up to the next keyframe. If at the last keyframe, update to the end of the trajectory
			start_idx = self.keyframes[kf_idx].id + 1
			end_idx = self.keyframes[kf_idx + 1].id if kf_idx + 1 < len(self.keyframes) else len(self.frames)
			prev = self.keyframes[kf_idx].pose
			for idx in range(start_idx, end_idx):
				frame = self.frames[idx]
				prev = np.dot(prev, np.linalg.inv(frame.relative_pose))
				frame.pose = prev

