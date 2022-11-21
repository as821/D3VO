from optimizer import Map
from frontend import Frame, Point, match_frame_kps
from depth_pose_net import Networks
import numpy as np

class D3VO:
    def __init__(self, intrinsic):
        self.intrinsic = intrinsic
        self.mp = Map()
        self.nn = Networks()

    def process_frame(self, frame, optimize=False):
        """Process a single frame with D3VO. Pass through DepthNet/PoseNet, frontend tracking, 
        and backend optimization (if optimize == True)."""
        # TODO run D3VO DepthNet and PoseNet (using Monodepth2 networks as placeholders)
        frame_shape = frame.shape[:2][::-1]
        np.random.seed(100)           # use same seed for debugging
        uncertainty = 100 * np.random.rand(*frame_shape)     # drop image channels
        brightness_params = (0, 0)      # a, b

        # Run DepthNet to get depth map
        depth = self.nn.depth(frame)

        if len(self.mp.frames) == 0:
            # Set first frame pose to identity rotation and no translation. Uses homogenous 4x4 matrix
            pose = np.eye(4)
        else:
            # Pass PoseNet the two most recent frames 
            pose = self.nn.pose(self.mp.frames[-1].image, frame) #np.random.rand(1) * np.eye(4)

        # Run frontend tracking
        if not self.frontend(frame, depth, uncertainty, pose, brightness_params):
            return

        # Run backend optimization
        if optimize:
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
        l1, l2 = match_frame_kps(f, prev_f)

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

        # TODO should we also be handling unmatched points in case they show up in later frames?? --> probably not, this is effectively loop closure

        return True
