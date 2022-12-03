import sys
import numpy as np



def numpy_to_kitti(pose):
    """Given a numpy 4x4 array, convert it into the KITTI .txt string format"""
    assert pose.shape == (4, 4)
    out = ""
    for i in range(3):      # exclude last row of homogenous matrix
        for j in range(pose.shape[1]):
            out += f"{pose[i, j]} "
    if out[-1] == " ":
        out = out[:-1]
    return out



def relative_to_global(poses):
    """Convert relative poses to global poses"""
    global_poses = []
    for idx, p in enumerate(poses):
        if idx == 0:
            global_poses.append(p)
        elif idx == 1:
            global_poses.append(np.linalg.inv(p))
        else:
            global_poses.append(np.dot(global_poses[-1], np.linalg.inv(p)))
    return global_poses



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("error: expecting 2 commandline arguments: source .npy and destination .txt files")
    

    pred_poses_np = np.load(sys.argv[1])
    kitti = [numpy_to_kitti(p) for p in pred_poses_np]

    # TODO write to output file
    with open(sys.argv[2], "w") as fd:
        for line in kitti:
            fd.write(f"{line}\n")
    print("Complete.")


