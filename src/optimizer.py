from frontend import Point, Frame
from helper import project, unproject

import g2o



class Map:
    """Class to store all current frames, points, etc."""
    def __init__(self):
        self.frames = []
        self.points = []
        self.frame_idx = self.pt_idx = 0

    def add_frame(self, frame):
        # TODO assumes no frame removal in ID assignment
        assert (type(frame) == Frame)
        ret = self.frame_idx
        self.frame_idx += 1
        self.frames.append(frame)
        return ret

    def add_point(self, pt):
        # TODO assumes no point removal in ID assignment
        assert (type(pt) == Point)
        ret = self.pt_idx
        self.pt_idx += 1
        self.points.append(pt)
        return ret

    def optimize(self):
        # Set up frames as vertices
        for f in self.frames:
            v_se3 = g2o.VertexSE3()
            v_se3.set_id(f.id)
            v_se3.set_estimate(f.pose)
            v_se3.set_fixed(f.id == 0)      # TODO when to make a frame pose fixed?? (ex. first frame, etc.)

        # Set up points as edges between frame vertices
        for p in self.points:
            for f in p.frames:
                pass




