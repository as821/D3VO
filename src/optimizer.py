from frontend import Point, Frame
from helper import project, unproject

import numpy as np
import g2o


def invert_depth(x):
	assert len(x) == 3 and x[2] != 0
	return np.array([x[0], x[1], 1]) / x[2]


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

	def optimize(self, intrinsic, iter=20, verbose=True):
		"""Run hypergraph-based optimization over current Points and Frames. Work in progress..."""
		# create optimizer (TODO just following example, likely incorrect for D3VO)
		opt = g2o.SparseOptimizer()
		solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
		solver = g2o.OptimizationAlgorithmLevenberg(solver)
		opt.set_algorithm(solver)
		opt.set_verbose(verbose)

		# Initialize an optimizer
		#opt, opt_frames, opt_pts = orig_optim(self, opt, intrinsic)
		print("setting up optimization...", end=' ')
		opt, opt_frames, opt_pts = ba_anchored_depth_optim(self, opt, intrinsic)

		# run optimizer
		print("initializing optimization...", end=' ')
		opt.initialize_optimization()
		print("starting optimization...", end=' ')
		opt.optimize(iter)
		print("optimization complete...", end=' ')

		# store optimization results 
		for p in self.points:
			# optimization gives unprojected point in 3D
			est = invert_depth(opt_pts[p].estimate())[-1]
			#assert est >= 0
			if est < 0:
				print("ERROR: INVALID POINT DEPTH", est)
				est = 0
				print(opt_pts[p].estimate())
			p.update_host_depth(est)
	
		for f in self.frames:
			est = opt_frames[f].estimate()
			f.pose = np.eye(4)
			f.pose[:3, :3] = est.rotation().matrix()
			f.pose[:3, 3] = est.translation()
			#print(f.pose)
		print("optimization complete")




def ba_anchored_depth_optim(self, opt, intrinsic):
	"""Optimization from ba_anchored_inverse_depth_demo.py example. Sort of along the line of what we are looking for.
		Note: anchor is the first frame where a point is observed"""
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
		v_se3 = g2o.VertexSE3Expmap()
		v_se3.set_estimate(g2o.SE3Quat(f.pose[0:3, 0:3], f.pose[0:3, 3])) 	# use frame pose estimate as initialization
		v_se3.set_id(f.id * 2)			# even IDs only
		if f.id < 2:
			v_se3.set_fixed(True)       # Hold first frame constant
		opt.add_vertex(v_se3)
		opt_frames[f] = v_se3

	# set up point edges between frames and depths
	for p in self.points:
		# setup vertex for depth estimate
		pt = g2o.VertexSBAPointXYZ()
		pt.set_id(p.id * 2 + 1)		# odd IDs, no collisions with frame ID

		# unproject point with depth estimate onto 3D world using the host frame depth estimate
		host_f, host_uv = p.get_host_frame()
		host_depth_est = host_f.depth[host_uv[0]][host_uv[1]]
		est = unproject(host_uv, host_depth_est, intrinsic)		# unprojected XYZ point estimate of this Point from POV of its host Frame (its anchor)
		pt.set_estimate(invert_depth(est))			
		#pt.set_fixed(False)
		opt_pts[p] = pt
		opt.add_vertex(pt)

		# host frame connects to every edge involving this point
		for idx, f in enumerate(p.frames[1:]):
			idx += 1													# avoid off by one, skipping host frame
			edge = g2o.EdgeProjectPSI2UV() 								#g2o.EdgeSE3PointXYZDepth() or EdgeProjectXYZ2UV
			edge.resize(3)
			edge.set_vertex(0, pt)										# connect to depth estimate
			# edge.set_vertex(1, opt_frames[host_frame])					# connect to host frame
			# edge.set_vertex(2, opt_frames[f])							# connect to frame where point was observed
			
			edge.set_vertex(1, opt_frames[f])							# connect to frame where point was observed
			edge.set_vertex(2, opt_frames[host_f])					# connect to host frame
			
			# estimate is the pixel coord. in the target frame
			uv_coord = f.kps[p.idxs[idx]]
			edge.set_measurement(uv_coord)
			edge.set_information(np.eye(2))								# simplified setting, no weights so use identity
			edge.set_robust_kernel(g2o.RobustKernelHuber())
			edge.set_parameter_id(0, 0)
			opt.add_edge(edge)
	return opt, opt_frames, opt_pts



def sba_demo2_optim(self, opt, intrinsic):
	"""Try out example bundle adjustment optimizers"""
	opt_frames, opt_pts = {}, {}
	
	# add camera
	focal = intrinsic[0, 0]
	cx = intrinsic[0, 2]
	cy = intrinsic[1, 2]       
	assert intrinsic[0, 0] == intrinsic[1, 1]		# fx == fy
	# cam = g2o.CameraParameters(f, (cx, cy), 0)         
	# cam.set_id(0)
	# opt.add_parameter(cam)  


	# set up frames as vertices
	for f in self.frames:
		# # add frame to the optimization graph as an SE(3) pose
		# v_se3 = g2o.VertexSE3()
		# v_se3.set_estimate(g2o.SE3Quat(f.pose[0:3, 0:3], f.pose[0:3, 3]).Isometry3d()) 	# use frame pose estimate as initialization
		# v_se3.set_id(f.id * 2)			# even IDs only
		# if f.id == 0:
		# 	v_se3.set_fixed(True)       # Hold first frame constant
		# opt.add_vertex(v_se3)
		# opt_frames[f] = v_se3

		sbacam = g2o.SBACam(g2o.SE3Quat(f.pose[0:3, 0:3], f.pose[0:3, 3]))
		sbacam.set_cam(focal, focal, cx, cy, 0)

		v_cam = g2o.VertexCam()
		v_cam.set_id(f.id * 2)
		v_cam.set_estimate(sbacam)
		if f.id < 2:
			v_cam.set_fixed(True)
		opt.add_vertex(v_cam)
		opt_frames[f] = v_cam


	# set up point edges between frames and depths
	for p in self.points:
		# setup vertex for depth estimate
		pt = g2o.VertexSBAPointXYZ()
		pt.set_id(p.id * 2 + 1)		# odd IDs, no collisions with frame ID
		pt.set_marginalized(True)

		# unproject point with depth estimate onto 3D world using the host frame depth estimate
		host_frame, host_uv_coord = p.get_host_frame()
		host_depth_est = host_frame.depth[int(host_uv_coord[0])][int(host_uv_coord[1])]
		est = unproject(host_uv_coord, host_depth_est, intrinsic)
		pt.set_estimate(est)			# --> set estimate to our current estimate of the true (x, y, z) coord. of this point
		
		#pt.set_fixed(False)
		opt_pts[p] = pt
		opt.add_vertex(pt)

		# host frame connects to every edge involving this point
		for idx, f in enumerate(p.frames[1:]):
			idx += 1													# avoid off by one, skipping host frame
			edge = g2o.EdgeProjectP2MC() 								#g2o.EdgeSE3PointXYZDepth() or EdgeProjectXYZ2UV
			#edge.resize(3)
			edge.set_vertex(0, pt)										# connect to depth estimate
			edge.set_vertex(1, opt_frames[f])
			#edge.set_vertex(1, opt_frames[host_frame])					# connect to host frame
			#edge.set_vertex(2, opt_frames[f])							# connect to frame where point was observed

			# get pixel coordinate of pt in frame f
			uv_coord = f.kps[p.idxs[idx]]
			edge.set_measurement(uv_coord)
			edge.set_information(np.eye(2))								# simplified setting, no weights so use identity
			edge.set_robust_kernel(g2o.RobustKernelHuber())
			#edge.set_parameter_id(0, 0)
			opt.add_edge(edge)

	return opt, opt_frames, opt_pts
	


def simple_optim(self, opt, intrinsic):
	"""Simple experimental optimizer, just try to get something to improve over PoseNet-only 30 frame error (0.005, std: 0.004). Current 30 frame error: 0.069, std: 0.036"""
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
		v_se3 = g2o.VertexSE3()
		v_se3.set_estimate(g2o.SE3Quat(f.pose[0:3, 0:3], f.pose[0:3, 3]).Isometry3d()) 	# use frame pose estimate as initialization
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
		for idx, f in enumerate(p.frames[1:]):
			idx += 1													# avoid off by one, skipping host frame
			edge = g2o.EdgeProjectPSI2UV() 								#g2o.EdgeSE3PointXYZDepth() or EdgeProjectXYZ2UV
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
	return opt, opt_frames, opt_pts


def orig_optim(self, opt, intrinsic):
	"""Original attempt at D3VO optimizer. Error at 30 frames: 0.085, std: 0.077"""
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
	return opt, opt_frames, opt_pts