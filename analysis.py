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

    def compute(self, landmarks):
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
        lknee = landmarks[25].y
        rknee = landmarks[26].y
        if lankle < lhips and rankle < rhips:
            return True
        return False
        
