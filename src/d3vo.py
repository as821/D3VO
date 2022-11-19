from optimizer import Map
from frontend import Frame, Point, match_frame_kps

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

        # Process f and the preceeding frame with a feature matcher. Iterate over match indices
        prev_f = self.mp.frames[-2]
        for idx1, idx2 in zip(*match_frame_kps(f, prev_f)):
            if idx2 in prev_f.pts:
                # Point already exists in prev_f
                prev_f.pts[idx2].add_observation(f, idx1)
            else:
                # New point
                pt = Point(self.mp)
                pt.add_observation(f, idx1)
                pt.add_observation(prev_f, idx2)

        # TODO should we also be handling unmatched points in case they show up in later frames??

        return True
