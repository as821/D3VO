import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import time

def homogenous(pt):
    assert pt.shape == (3,)
    return np.concatenate((pt, np.array([1])))


#plt.ion()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.draw()

def display_trajectory(poses):
    # poses are relative poses, must compose them
    xdata, ydata, zdata = [0], [0], [0]
    pt = np.array([0, 0, 0])
    for pose in poses:
        pt = (pose @ homogenous(pt))[:3]
        xdata.append(pt[0])
        ydata.append(pt[1])
        zdata.append(pt[2])

    # dotted lines between pose points
    for i in range(len(xdata)-1):
        plt.plot(xdata[i:i+2], ydata[i:i+2], zdata[i:i+2], 'bo', linestyle="--")

    ax.scatter3D(xdata, ydata, zdata, c='r')
    plt.draw()

    