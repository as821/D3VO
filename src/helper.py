import numpy as np
import cv2

def project(pt, intrinsic):
    """Project the specified 3D point onto 2D camera plane using known camera intrinsics"""
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    us = fx * pt.x / pt.z + cx
    vs = fy * pt.y / pt.z + cy
    return np.array(us, vs)
    


def unproject(pt, depth, intrinsic):
    """Unproject a given point in 2D into 3D space using known camera intrinsics"""
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    xs = (pt[0] - cx) / fx * depth
    ys = (pt[1] - cy) / fy * depth
    return np.array([xs, ys, depth])



def calc_avg_matches(frame, out_frame, show_correspondence=False):
    """Return the average number of matches each keypoint in the specified frame has. 
    Visualize these matches if show_correspondence=True"""
    n_match = 0		# avg. number of matches of keypoints in the current frame
    for idx in frame.pts:
        # red line to connect current keypoint with Point location in other frames
        pt = [int(i) for i in frame.kps[idx]]
        if show_correspondence:
            for f, f_idx in zip(frame.pts[idx].frames, frame.pts[idx].idxs):
                cv2.line(out_frame, pt, [int(i) for i in f.kps[f_idx]], (0, 0, 255), thickness=2)
        n_match += len(frame.pts[idx].frames)
    if len(frame.pts) > 0:
        n_match /= len(frame.pts)
    return n_match, out_frame


