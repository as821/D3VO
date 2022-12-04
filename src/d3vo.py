from optimizer import Map
from frontend import Frame, Point, match_frame_kps
from depth_pose_net import Networks
import numpy as np
from copy import deepcopy

class D3VO:
	def __init__(self, intrinsic, trajectory_scale=27):
		self.intrinsic = intrinsic
		self.mp = Map()
		self.nn = Networks()
		self.trajectory_scale = trajectory_scale

	def process_frame(self, frame, optimize=True):
		"""Process a single frame with D3VO. Pass through DepthNet/PoseNet, frontend tracking, 
		and backend optimization (if optimize == True)."""
		# TODO run D3VO DepthNet and PoseNet (using Monodepth2 networks as placeholders)
		np.random.seed(100)           # use same seed for debugging
		uncertainty = np.zeros_like(frame)		# uncertainty == 0, get weight of 1. as uncertainty increases (positive or negative), weight drops
		brightness_params = (0, 0)      # a, b

		# Run DepthNet to get depth map
		depth = self.nn.depth(frame)

		if len(self.mp.frames) == 0:
			# Set first frame pose to identity rotation and no translation. Uses homogenous 4x4 matrix
			pose = np.eye(4)
		else:
			# Pass PoseNet the two most recent frames 
			pose = self.nn.pose(self.mp.frames[-1].image, frame)

		# Run frontend tracking
		if not self.frontend(frame, depth, uncertainty, pose, brightness_params):
			return

		# Run backend optimization
		if optimize:
			self.mp.optimize(self.intrinsic)


	def frontend(self, frame, depth, uncertainty, pose, brightness_params):
		"""Run frontend tracking on the given frame --> just process every frame with a basic feature extractor for right now.
		Return true to continue onto backend optimization"""
		# create frame and add it to the map
		f = Frame(self.mp, frame, depth, uncertainty, pose, brightness_params)

		# cannot match first frame to any previous frames (but make it a keyframe)
		if f.id == 0:
			self.mp.check_add_key_frame(f)
			return False

		# TODO this should be done with DSO's feature extractor/matching approach, this is just to enable backend work

		# Process f and the preceeding frame with a feature matcher. Iterate over match indices
		prev_f = self.mp.frames[-2]
		l1, l2 = match_frame_kps(f, prev_f)

		# Store matches
		for idx1, idx2 in zip(l1, l2):
			if idx2 in prev_f.pts:
				# Point already exists in prev_f
				prev_f.pts[idx2].add_observation(f, idx1)
			else:
				# New point
				pt = Point(self.mp)
				pt.add_observation(f, idx1)
				pt.add_observation(prev_f, idx2)

		# Check if this new frame should be a keyframe
		if self.mp.check_add_key_frame(f):
			# Keyframe has been added, run backend optimization
			return True

		return False


	def relative_to_global(self):
		"""Convert relative pose stored in frames into a global pose."""
		pred_pose = []
		for idx, f in enumerate(self.mp.frames[1:]):		
			if idx > 1:
				pred_pose.append(np.dot(pred_pose[idx-1], np.linalg.inv(self.mp.frames[idx].pose)))
			else:
				pred_pose.append(np.linalg.inv(f.pose))

		for t in range(len(pred_pose)):
			pred_pose[t][:3, 3] *= self.trajectory_scale
		return pred_pose


	def run_eval(self, gt_pose, eval, plot_traj=False):
		"""Evaluate the performance of D3VO compared to the ground truth poses provided and print result."""
		# recompute global poses from stored relative poses every time to allow bundle adjustment changes to propagate
		pred_pose = {idx+1 : p for idx, p in enumerate(self.relative_to_global())}
		ate = eval.compute_ATE(gt_pose, pred_pose)
		rpe_trans, rpe_rot = eval.compute_RPE(gt_pose, pred_pose)
		print("ATE (m): ", ate, ". RPE (m): ", rpe_trans, ". RPE (deg): ", rpe_rot * 180 /np.pi)

		if plot_traj:
			eval.plot_trajectory(gt_pose, pred_pose, 9)

