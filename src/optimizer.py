from frontend import Point, Frame
from helper import project, unproject

import numpy as np
import g2o



class Map:
	"""Class to store and optimize over all current frames, points, etc."""
	def __init__(self):
		self.frames = []
		self.points = []
		self.frame_idx = self.pt_idx = 0

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


		# set up frames as vertices
		for f in self.frames:
			# add frame to the optimization graph as an SE(3) pose
			v_se3 = g2o.VertexD3VOFramePose(f.image)
			v_se3.set_estimate(g2o.SE3Quat(f.pose[0:3, 0:3], f.pose[0:3, 3])) 	# .Isometry3d() use frame pose estimate as initialization
			v_se3.set_id(f.id * 2)			# even IDs only
			if f.id < 2:
				v_se3.set_fixed(True)       # Hold first frame constant
			opt.add_vertex(v_se3)
			opt_frames[f] = v_se3

		# set up point edges between frames and depths
		for p in self.points:
			# setup vertex for depth estimate
			host_frame, host_uv_coord = p.get_host_frame()
			pt = g2o.VertexD3VOPointDepth(host_uv_coord[0], host_uv_coord[1])       
			pt.set_id(p.id * 2 + 1)		# odd IDs, no collisions with frame ID

			# give optimizer initial depth estimates for all pixels in the DSO pixel pattern
			# pix_pattern, success = p.pixel_pattern(host_frame.image.shape[0], host_frame.image.shape[1])
			# if not success:
			# 	# pixel in pattern was out of bounds of the image
			# 	continue
			# host_depth_est = [host_frame.depth[uv[0]][uv[1]] for uv in pix_pattern]


			host_depth_est = host_frame.depth[host_uv_coord[0]][host_uv_coord[1]]

			pt.set_estimate(host_depth_est)
			pt.set_fixed(False)
			opt_pts[p] = pt
			opt.add_vertex(pt)

			# host frame connects to every edge involving this point
			for idx, f in enumerate(p.frames[1:]):
				idx += 1													# avoid off by one, skipping host frame
				edge = g2o.EdgeProjectD3VO() 								
				edge.resize(3)
				edge.set_vertex(0, pt)										# connect to depth estimate
				edge.set_vertex(1, opt_frames[host_frame])					# connect to host frame
				edge.set_vertex(2, opt_frames[f])							# connect to frame where point was observed
				
				edge.set_information(np.eye(3))								# simplified setting, no weights so use identity
				edge.set_robust_kernel(g2o.RobustKernelHuber())
				edge.set_parameter_id(0, 0)
				opt.add_edge(edge)

		# run optimizer
		opt.initialize_optimization()
		opt.optimize(iter)

		# store optimization results 
		for p in self.points:
			# optimization gives unprojected point in 3D
			est = opt_pts[p].estimate()
			#assert est >= 0
			#p.update_host_depth(est)
			# print(est)
	
		for f in self.frames:
			est = opt_frames[f].estimate()
			f.pose = np.eye(4)
			f.pose[:3, :3] = est.rotation().matrix()
			f.pose[:3, 3] = est.translation()
			#print(f.pose)
		return
		

