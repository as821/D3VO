from frontend import Point, Frame, match_frame_kps
from helper import project, unproject

import numpy as np
import g2o



class Map:
	"""Class to store and optimize over all current frames, points, etc."""
	def __init__(self, N_f=7):
		self.frames = []
		self.key_frames = []
		self.points = []
		self.frame_idx = self.pt_idx = 0
		self.N_f = N_f

	def set_N_f(self, N_f):
		self.N_f = N_f

	def check_key_frame(self, frame, instrinsics):
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
		R1 = (np.linalg.inv(instrinsics) @ last_key_frame.pose)[:3, :3]
		R2 = (np.linalg.inv(instrinsics) @ frame.pose)[:3, :3]
		homography_t = instrinsics @ R1 @ np.linalg.inv(R2) @ np.linalg.inv(instrinsics)

		f = 0
		ft = 0
		a = 0

		for idx1, idx2 in zip(l1, l2):
			x1, y1 = last_key_frame.kps[idx1].pt
			x2, y2 = frame.kps[idx2].pt
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

	def marginalize(self):
		latest_key_frame = self.key_frames[-1]
		max_dist = 0
		max_dist_idx = 0
		marginalized_count = 0

		# Can marginalize everything apart from the last two keyframes:
		for i in range(len(self.key_frames) - 1):
			l1, l2 = match_frame_kps(last_key_frame, self.key_frames[i])
			if len(l2) / len(self.key_frames[i].kps) < 0.1:
				self.key_frames[i].marginalize = True
				marginalized_count += 1
			frame_dist = np.linalg.norm(latest_key_frame - self.key_frames[i])
			if frame_dist > max_dist:
				max_dist = frame_dist
				max_dist_idx = i

		if len(self.key_frames) > self.N_f and marginalized_count == 0:
			self.key_frames[max_dist_idx].marginalize = True

	def add_frame(self, frame, instrinsics):
		"""Add a Frame to the Map"""
		# TODO assumes no frame removal in ID assignment
		assert (type(frame) == Frame)
		if len(self.frames) > 0:
			key_frame = self.check_key_frame(frame, instrinsics)
		else:
			key_frame = True

		if key_frame:
			self.key_frames.append(frame)
			self.marginalize()

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

	def optimize(self, intrinsic, iter=10):
		"""Run hypergraph-based optimization over current Points and Frames. Work in progress..."""
		# create optimizer (TODO just following example, likely incorrect for D3VO)
		opt = g2o.SparseOptimizer()
		solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
		solver = g2o.OptimizationAlgorithmLevenberg(solver)
		opt.set_algorithm(solver)
		opt.set_verbose(True)

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
		for f in self.frames:
			# add frame to the optimization graph as an SE(3) pose
			init_pose = f.pose
			v_se3 = g2o.VertexSE3Expmap()
			v_se3.set_estimate(g2o.SE3Quat(init_pose[0:3, 0:3], init_pose[0:3, 3])) 	# use frame pose estimate as initialization
			v_se3.set_id(f.id * 2)			# even IDs only
			if f.id == 0:
				v_se3.set_fixed(True)       # Hold first frame constant
			opt.add_vertex(v_se3)
			opt_frames[f] = v_se3

		# set up point edges between frames and depths
		for p in self.points:
			# setup vertex for depth estimate
			pt = g2o.VertexPointXYZ()
			pt.set_id(p.id * 2 + 1)		# odd IDs, no collisions with frame ID

			# unproject point with depth estimate onto 3D world using the host frame depth estimate
			host_frame, host_uv_coord = p.get_host_frame()
			host_depth_est = host_frame.depth[int(host_uv_coord[0])][int(host_uv_coord[1])]
			est = unproject(host_uv_coord, host_depth_est, intrinsic)
			pt.set_estimate(est)			
			
			pt.set_fixed(False)
			opt_pts[p] = pt
			opt.add_vertex(pt)

			# host frame connects to every edge involving this point
			for idx, f  in enumerate(p.frames[1:]):
				idx += 1													# avoid off by one, skipping host frame
				edge = g2o.EdgeProjectPSI2UV()								# or EdgeProjectXYZ2UV??
				edge.resize(3)
				edge.set_vertex(0, pt)										# connect to depth estimate
				edge.set_vertex(1, opt_frames[host_frame])					# connect to host frame
				edge.set_vertex(2, opt_frames[f])							# connect to frame where point was observed
				uv_coord = f.kps[p.idxs[idx]]
				#inten = f.image[uv_coord[1], uv_coord[0]]
				#edge.set_measurement(inten)		# measurement is host frame pixel intensity (u/v coordinate swap)
				
				# TODO this seems incorrect
				edge.set_measurement(uv_coord)
				
				edge.set_information(np.eye(2))								# simplified setting, no weights so use identity
				edge.set_robust_kernel(g2o.RobustKernelHuber())
				edge.set_parameter_id(0, 0)
				opt.add_edge(edge)

		# run optimizer
		opt.initialize_optimization()
		opt.optimize(iter)

		# store optimization results 
		for p in self.points:
			# optimization gives unprojected point in 3D
			est = opt_pts[p].estimate()[-1]
			assert est >= 0
			p.update_host_depth(est)
			# print(est)
	
		for f in self.frames:
			est = opt_frames[f].estimate()
			f.pose = np.eye(4)
			f.pose[:3, :3] = est.rotation().matrix()
			f.pose[:3, 3] = est.translation()
			print(f.pose)
		return
		

