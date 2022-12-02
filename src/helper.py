import numpy as np
import cv2
import os

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


# from monodepth2 evalute_pose
def evaluate_pose(gt_path, pred_poses):
    """Given a path to a file with ground truth poses, evaluate the performance so far. 
    If partial=True, only evaluate # of frames == # of predicted poses provided."""
    gt_poses_path = os.path.join(gt_path)
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate((gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]

    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(
            np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    # NOTE: this uses a "track length" of 1 vs. monodepth2 which uses a track length of 5 (effectively
    # a smoothing operation). Take this into account when comparing results
    ates = []
    num_frames = gt_xyzs.shape[0]
    assert len(pred_poses) <= num_frames-1
    iter_len = min(len(pred_poses), num_frames-1)
    for i in range(0, iter_len):
        local_xyzs = np.array(dump_xyz([pred_poses[i]]))
        gt_local_xyzs = np.array(dump_xyz([gt_local_poses[i]]))
        ates.append(compute_ate(gt_local_xyzs, local_xyzs))
    return ates


# from monodepth2 evalute_pose
def evaluate_global_pose(gt_path, pred_poses):
    """Given a path to a file with ground truth poses, evaluate the performance so far. 
    If partial=True, only evaluate # of frames == # of predicted poses provided."""
    gt_poses_path = os.path.join(gt_path)
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate((gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]

    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(
            np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    # NOTE: this uses a "track length" of 1 vs. monodepth2 which uses a track length of 5 (effectively
    # a smoothing operation). Take this into account when comparing results
    ates = []
    num_frames = gt_xyzs.shape[0]
    assert len(pred_poses) <= num_frames-1
    iter_len = min(len(pred_poses), num_frames-1)
    for i in range(0, iter_len):
        local_xyzs = np.array(dump_xyz_global([pred_poses[i]]))
        gt_local_xyzs = np.array(dump_xyz([gt_local_poses[i]]))
        ates.append(compute_ate(gt_local_xyzs, local_xyzs))
    return ates


# from monodepth2 evalute_pose, from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


def dump_xyz_global(source_to_target_transformations):
    # dump_xyz, but treat poses as global rather than relative poses
    xyzs = []
    xyzs.append(np.eye(4)[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        xyzs.append(source_to_target_transformation[:3, 3])
    return xyzs



# from monodepth2 evalute_pose, from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):
    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse
