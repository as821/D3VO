import sys
import cv2
import os
import numpy as np
import argparse

from d3vo import D3VO

sys.path.insert(1, os.path.join(sys.path[0], '../kitti-odom-eval-master'))
from kitti_odometry import KittiEvalOdom


DEBUG = False
PER_FRAME_ERROR = True


def offline_vo(cap, weights_path, gt_path, save_path, out_dir):
	"""Run D3VO on offline video"""
	intrinsic = np.array([[F,0,W//2,0],[0,F,H//2,0],[0,0,1,0]])

	d3vo = D3VO(weights_path, intrinsic)

	if gt_path != "":
		# Use open source KITTI evaluation code, requires poses in form of a dictionary
		eval = KittiEvalOdom(out_dir)
		gt_poses = eval.load_poses_from_txt(gt_path)

	# Run D3VO offline with prerecorded video
	i = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True:
			frame = cv2.resize(frame, (W, H))
			print("\n*** frame %d/%d ***" % (i, CNT))
			d3vo.process_frame(frame)

			# Run evaluation
			if gt_path != "" and PER_FRAME_ERROR and len(d3vo.mp.frames) > 1:
				d3vo.run_eval(gt_poses, eval, plot_traj=(len(d3vo.mp.frames) % 10 == 0))
		else:
			break
		i += 1

		if DEBUG:
			cv2.imshow('d3vo', frame)
			if cv2.waitKey(1) == 27:     # Stop if ESC is pressed
				break
	
	# Final trajectory evaluation
	if gt_path != "":
		d3vo.run_eval(gt_poses, eval, plot_traj=True)


	# Store pose predictions to a file (do not save identity pose of first frame)
	save_path = os.path.join(save_path)
	np.save(save_path, d3vo.mp.relative_to_global())
	print("-> Predictions saved to", save_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("path", type=str, help="path to the input video")
	parser.add_argument("weights", type=str, help="path to directory containing trained DepthNet and PoseNet weights")
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
	offline_vo(cap, args.weights, args.gt, args.save, args.out)




