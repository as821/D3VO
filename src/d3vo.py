from optimizer import Map
from frontend import Frame, Point, match_frame_kps

import numpy as np

class D3VO:
    def __init__(self, intrinsic):
        self.intrinsic = intrinsic
        self.mp = Map()


    def process_frame(self, frame):
        # TODO run DepthNet and PoseNet (these are placeholders)
        frame_shape = frame.shape[:2][::-1]
        np.random.seed(100)           # use same seed for debugging
        depth = uncertainty = 100 * np.random.rand(*frame_shape)     # drop image channels
        brightness_params = (0, 0)      # a, b

        if len(self.mp.frames) == 0:
            # Set first frame pose to identity
            pose = np.concatenate((np.eye(3), np.zeros(shape=(3, 1))), axis=1)  # identity rotation, no translation
        else:
            # TODO pose net here, inject some noise until then
            pose = np.random.rand(1) * np.concatenate((np.eye(3), np.zeros(shape=(3, 1))), axis=1)


        # Run frontend tracking
        if not self.frontend(frame, depth, uncertainty, pose, brightness_params):
            return

        # Run backend optimization
        self.mp.optimize(self.intrinsic)


    def frontend(self, frame, depth, uncertainty, pose, brightness_params):
        """Run frontend tracking on the given frame --> just process every frame with a basic feature extractor for right now.
        Return true to continue onto backend optimization"""
        # create frame and add it to the map
        f = Frame(self.mp, frame, depth, uncertainty, pose, brightness_params)

        # cannot match first frame to any previous frames
        if f.id == 0:
            return False

        # TODO this should be done with DSO's feature extractor/matching approach, this is just to enable backend work

        # Process f and the preceeding frame with a feature matcher. Iterate over match indices
        prev_f = self.mp.frames[-2]
        l1, l2, pose = match_frame_kps(f, prev_f)

        # Store matches
        for idx1, idx2 in zip(l1, l2):
            if idx2 in prev_f.pts:
                # Point already exists in prev_f
                prev_f.pts[idx2].add_observation(f, idx1)
            else:
                # New point
                pt = Point(self.mp)
                pt.add_observation(f, idx1)
                pt.add_observation(prev_f, idx2)

        # TODO Get a better initial pose estimate (basic approach for now)
        f.pose = pose 

        # run optimization every 5 frames
        if f.id % 5 != 0:
            return False
        

        # TODO should we also be handling unmatched points in case they show up in later frames?? --> probably not, this is loop closure

        return True
