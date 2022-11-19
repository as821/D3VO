import numpy as np


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

