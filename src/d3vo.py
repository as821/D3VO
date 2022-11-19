from optimizer import Map
from frame_point import Frame, Point
from frontend import match_frames

class D3VO:
    def __init__(self, intrinsic):
        self.intrinsic = intrinsic
        self.mp = Map()


    def process_frame(self, frame):
        # Run frontend tracking
        if not self.frontend(frame):
            return

        # Run backend optimization
        self.mp.optimize()


    def frontend(self, frame):
        """Run frontend tracking on the given frame --> just process every frame with a basic feature extractor for right now.
        Return true to continue onto backend optimization"""
        # create frame and add it to the map
        f = Frame(self.mp, frame, None, None, None, None)

        # cannot match first frame to any previous frames
        if f.id == 0:
            return False

        # TODO this should be done with DSO's feature extractor/matching approach, this is just to enable backend work
        # Process the frame with a feature extractor + matcher
        #match_frames(f, self.mp.frames[-2])
        #return True

        return False
