from frame_point import Point, Frame
from helper import project, unproject

import g2o



class Map:
    """Class to store all current frames, points, etc."""
    def __init__(self):
        self.keyframes = []
        self.points = []

    def add_keyframe(self):
        pass

    def add_point(self):
        pass


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




