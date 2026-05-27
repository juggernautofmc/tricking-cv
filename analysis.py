import numpy as np

# PURPOSE: compute derived athletic metrics from raw landmarks
# - input: the landmark data object from pose.py
# - output: a set of metrics useful for tricking/karate coaching, e.g.
#     - joint angles (knee, hip, elbow) in degrees
#     - how high each foot is relative to the body
#     - whether the athlete is airborne (both feet off the ground)
# - this file should NOT know anything about video frames or drawing —
#   it only does math on landmark coordinates

## LANDMARK INDICES
LEFT_SHOULDER_ID = 11
RIGHT_SHOULDER_ID = 12
LEFT_ELBOW_ID = 13
RIGHT_ELBOW_ID = 14
LEFT_WRIST_ID = 15
RIGHT_WRIST_ID = 16
LEFT_HIP_ID = 23
RIGHT_HIP_ID = 24
LEFT_KNEE_ID = 25
RIGHT_KNEE_ID = 26
LEFT_HEEL_ID = 29
RIGHT_HEEL_ID = 30

class Analysis:
    def __init__(self):
        # input: none, output: none, useless function lol 
        self.groundheight = None
        self.frame_count = 0 #yup we tracking frames now

    def compute(self, landmarks):
        self.frame_count += 1
        if self.frame_count <= 30: 
            lankle = landmarks.landmark[27].y
            rankle = landmarks.landmark[28].y
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

        if self.frame_count > 30: #we waiting until 30 frames, assume they haven't jumped before then
            if landmarks is None:
                return False
            lhips = landmarks.landmark[23].y 
            rhips = landmarks.landmark[24].y
            lankle = landmarks.landmark[27].y
            rankle = landmarks.landmark[28].y
            if lankle < lhips and rankle < rhips:
                return True
            if lankle < self.groundheight or rankle < self.groundheight:
                return True
        return False
            
