import numpy as np

# PURPOSE: compute derived athletic metrics from raw landmarks
# - input: the landmark data object from pose.py
# - output: a set of metrics useful for tricking/karate coaching, e.g.
#     - joint angles (knee, hip, elbow) in degrees
#     - how high each foot is relative to the body
#     - whether the athlete is airborne (both feet off the ground)
# - this file should NOT know anything about video frames or drawing —
#   it only does math on landmark coordinates


class Analysis:
    def __init__(self):
        # input: none, output: none, useless function lol 
        self.groundheight = None
        self.frame_count = 0 #yup we tracking frames now

    def compute(self, landmarks):
        self.frame_count += 1
        if self.frame_count <= 30: 
            lankle = landmarks[27].y
            rankle = landmarks[28].y
            if self.groundheight is None:
                self.groundheight = min(lankle, rankle) # we want min since larger y means closest to bottom of screen
            else:
                self.groundheight = min(self.groundheight, lankle, rankle)

        # input: 33 landmarks from pose.py, output: dict of metrics (joint angles, foot height, airborne)
        pass

    def joint_angle(self, a, b, c):
        # input: three landmarks (a, b, c) where b is the joint, output: angle at b in degrees
        pass

    def is_airborne(self, landmarks):
        # input: 33 landmarks, output: True if both feet are off the ground, False otherwise
        if landmarks is None:
            return False
        lhips = landmarks[23].y 
        rhips = landmarks[24].y
        lankle = landmarks[27].y
        rankle = landmarks[28].y
        if lankle < lhips and rankle < rhips:
            return True
        return False
        
