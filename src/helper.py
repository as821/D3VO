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




import os


### TODO just helpers for now
def poseRt(R, t):
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t
  return ret

# pose
def fundamentalToRt(F):
  W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
  U,d,Vt = np.linalg.svd(F)
  if np.linalg.det(U) < 0:
    U *= -1.0
  if np.linalg.det(Vt) < 0:
    Vt *= -1.0
  R = np.dot(np.dot(U, W), Vt)
  if np.sum(R.diagonal()) < 0:
    R = np.dot(np.dot(U, W.T), Vt)
  t = U[:, 2]

  # TODO: Resolve ambiguities in better ways. This is wrong.
  if t[2] < 0:
    t *= -1
  
  return np.linalg.inv(poseRt(R, t))

