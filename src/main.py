import sys
import cv2
import os
import numpy as np

from d3vo import D3VO

from display import display_trajectory


def offline_slam(cap):
	"""Run D3VO on offline video"""
	intrinsic = np.array([[F,0,W//2,0],[0,F,H//2,0],[0,0,1,0]])
	d3vo = D3VO(intrinsic)

	# Run D3VO offline with prerecorded video
	i = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True:
			frame = cv2.resize(frame, (W, H))
			print("\n*** frame %d/%d ***" % (i, CNT))
			d3vo.process_frame(frame)

			if DEBUG:
				# plot all poses (invert so they move in correct direction)
				plt = [d3vo.mp.frames[0].pose]
				for f in d3vo.mp.frames[1:]:
					plt.append(np.linalg.inv(f.pose))
				display_trajectory(plt)

				# show keypoints with matches in this frame
				n_match = 0		# avg. number of matches of keypoints in the current frame
				for idx in d3vo.mp.frames[-1].pts:
					# green for keypoint in this frame
					pt = d3vo.mp.frames[-1].kps[idx]
					cv2.circle(frame, [int(i) for i in pt], color=(0, 255, 0), radius=3)

					# red line to connect current keypoint with Point location in other frames
					for f, f_idx in zip(d3vo.mp.frames[-1].pts[idx].frames, d3vo.mp.frames[-1].pts[idx].idxs):
						f_pt = f.kps[f_idx]
						cv2.line(frame, [int(i) for i in pt], [int(i) for i in f_pt], (0, 0, 255), thickness=2)
					n_match += len(d3vo.mp.frames[-1].pts[idx].frames)
				if len(d3vo.mp.frames[-1].pts) > 0:
					n_match /= len(d3vo.mp.frames[-1].pts)

				print("Matches: %d / %d (%f)" % (len(d3vo.mp.frames[-1].pts), len(d3vo.mp.frames[-1].kps), n_match))

		else:
			break
		i += 1

		if DEBUG:
			cv2.imshow('d3vo', frame)
			if cv2.waitKey(1) == 27:     # Stop if ESC is pressed
				break


if __name__ == "__main__":
	# Requires path the video file
	if len(sys.argv) < 2:
		print("insufficient number of arguments, expecting path to an .mp4 file")
		exit(-1)

	cap = cv2.VideoCapture(sys.argv[1])

	# camera parameters from video (offline, using pre-recorded video)
	W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	# allow use of environment variable settings for focal length, seeking
	F = float(os.getenv("F", default="984"))
	DEBUG = bool(os.getenv("D", default="True"))
	if os.getenv("SEEK") is not None:
		cap.set(cv2.CAP_PROP_POS_FRAMES, int(os.getenv("SEEK")))

	# resize video if needed
	if W > 1024:
		downscale = 1024.0/W
		F *= downscale
		H = int(H * downscale)
		W = 1024
	print("using camera %dx%d with F %f" % (W,H,F))

	offline_slam(cap)





