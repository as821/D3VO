import sys
import cv2
import os
import numpy as np
import argparse
from copy import deepcopy

from d3vo import D3VO
from helper import calc_avg_matches 

sys.path.insert(1, os.path.join(sys.path[0], '../kitti-odom-eval-master'))
from kitti_odometry import KittiEvalOdom


DEBUG = True
PER_FRAME_ERROR = True


def offline_vo(cap, gt_path, save_path, out_dir):
	"""Run D3VO on offline video"""
	intrinsic = np.array([[F,0,W//2,0],[0,F,H//2,0],[0,0,1,0]])

	d3vo = D3VO(intrinsic)

	if gt_path != "":
		# Use open source KITTI evaluation code. Monodepth2 evaluation code is incorrect
		eval = KittiEvalOdom(out_dir)
		gt_poses = eval.load_poses_from_txt(gt_path)
		pred_pose_kitti = {}

	# Run D3VO offline with prerecorded video
	i = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True:
			frame = cv2.resize(frame, (W, H))
			print("\n*** frame %d/%d ***" % (i, CNT))
			d3vo.process_frame(frame)

			if DEBUG:
				# show keypoints with matches in this frame
				for pidx, p in enumerate(d3vo.mp.frames[-1].kps):
					if pidx in d3vo.mp.frames[-1].pts:
						# green for matched keypoints 
						cv2.circle(frame, [int(i) for i in p], color=(0, 255, 0), radius=3)
					else:
						# black for unmatched keypoint in this frame
						cv2.circle(frame, [int(i) for i in p], color=(0, 0, 0), radius=3)

				# Calculate the average number of frames each point in the last frame is also visible in
				n_match, frame = calc_avg_matches(d3vo.mp.frames[-1], frame, show_correspondence=False)
				print("Matches: %d / %d (%f)" % (len(d3vo.mp.frames[-1].pts), len(d3vo.mp.frames[-1].kps), n_match))

			# Run evaluation
			if gt_path != "" and PER_FRAME_ERROR and len(d3vo.mp.frames) > 1:
				# if i > 1:
				# 	pred_pose_kitti[i] = np.dot(pred_pose_kitti[i-1], np.linalg.inv(d3vo.mp.frames[i].pose))
				# else:
				# 	pred_pose_kitti[i] = np.linalg.inv(d3vo.mp.frames[i].pose)

				# This is hacky, but our trajectory follows the same shape as the ground truth but has incorrect scale
				# This is likely an issue with the scale of DepthNet + PoseNet
				translation_scale = 27
				pred_pose_kitti = {}
				for idx, f in enumerate(d3vo.mp.frames[1:]):		# recompute global every time to allow bundle adjustment changes to propagate
					if idx > 1:
						pred_pose_kitti[idx] = np.dot(pred_pose_kitti[idx-1], np.linalg.inv(d3vo.mp.frames[idx].pose))
					else:
						pred_pose_kitti[idx] = np.linalg.inv(f.pose)

				for t in pred_pose_kitti:
					pred_pose_kitti[t][:3, 3] *= translation_scale

				ate = eval.compute_ATE(gt_poses, pred_pose_kitti)
				rpe_trans, rpe_rot = eval.compute_RPE(gt_poses, pred_pose_kitti)
				print("ATE (m): ", ate, ". RPE (m): ", rpe_trans, ". RPE (deg): ", rpe_rot * 180 /np.pi)

				if DEBUG and len(d3vo.mp.frames) % 10 == 0:
					eval.plot_trajectory(gt_poses, pred_pose_kitti, 9)
		else:
			break
		i += 1

		if DEBUG:
			cv2.imshow('d3vo', frame)
			if cv2.waitKey(1) == 27:     # Stop if ESC is pressed
				break
	
	# Final trajectory evaluation
	if gt_path != "":
		# Do not include identity pose of first frame in evaluation
		ate = eval.compute_ATE(gt_poses, pred_pose_kitti)
		rpe_trans, rpe_rot = eval.compute_RPE(gt_poses, pred_pose_kitti)
		print("ATE (m): ", ate, ". RPE (m): ", rpe_trans, ". RPE (deg): ", rpe_rot * 180 /np.pi)


	# Store pose predictions to a file (do not save identity pose of first frame)
	save_path = os.path.join(save_path)
	np.save(save_path, [f.pose for f in d3vo.mp.frames[1:]])
	print("-> Predictions saved to", save_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("path", type=str, help="path to the input video")
	parser.add_argument("--gt", type=str, default="", help="path to .txt file with ground truth poses")
	parser.add_argument("--save", type=str, default="poses.npy", help="path to output file to store pose predictions")
	parser.add_argument("--out", type=str, help="path to output directory to store plots")
	parser.add_argument("--focal", type=int, default=984, help="focal length of camera")
	args = parser.parse_args()

	cap = cv2.VideoCapture(args.path)

	# camera parameters from video (offline, using pre-recorded video)
	W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	F = args.focal

	# downsize video if needed
	if W > 1024:
		downscale = 1024.0/W
		F *= downscale
		H = int(H * downscale)
		W = 1024
	print("using camera %dx%d with F %f" % (W,H,F))

	# run offline visual odometry on provided video
	offline_vo(cap, args.gt, args.save, args.out)




