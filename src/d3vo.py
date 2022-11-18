from optimizer import Map



class D3VO:
    def __init__(self, intrinsic):
        self.intrinsic = intrinsic
        self.mp = Map()


    def process_frame(self, frame):
        # TODO run DepthNet/PoseNet (maybe only run these on keyframes? we'll see)


        # TODO run frontend tracking


        # TODO run backend optimization


        pass