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
        self.leftgroundheight = None
        self.rightgroundheight = None
        self.prevlefthips = None
        self.prevrighthips = None
        self.prevleftknee = None
        self.prevrightknee = None
        self.frame_count = 0 #yup we tracking frames now

    def compute(self, landmarks):
        self.frame_count += 1
        if self.frame_count <= 5: 
            lheel = landmarks[LEFT_HEEL_ID].y
            rheel = landmarks[RIGHT_HEEL_ID].y
            lhips = landmarks[LEFT_HIP_ID].y
            rhips = landmarks[RIGHT_HIP_ID].y
            lknee = landmarks[LEFT_KNEE_ID].y
            rknee = landmarks[RIGHT_KNEE_ID].y
            if self.leftgroundheight is None:
                self.leftgroundheight = lheel
                self.rightgroundheight = rheel
                self.prevlefthips = lhips
                self.prevrighthips = rhips
                self.prevleftknee = lknee
                self.prevrightknee = rknee
            #else:
                #self.groundheight = min(self.groundheight, lheel, rheel)

        # input: 33 landmarks from pose.py, output: dict of metrics (joint angles, foot height, airborne)
        pass

    def joint_angle(self, a, b, c):
        # input: three landmarks (a, b, c) where b is the joint, output: angle at b in degrees
        pass

    def is_airborne(self, landmarks):
        # input: 33 landmarks, output: True if both feet are off the ground, False otherwise

        if self.frame_count > 5: #we waiting until 30 frames, assume they haven't jumped before then
            if landmarks is None:
                return False
            lhips = landmarks[LEFT_HIP_ID].y
            rhips = landmarks[RIGHT_HIP_ID].y
            lheel = landmarks[LEFT_HEEL_ID].y
            rheel = landmarks[RIGHT_HEEL_ID].y
            lknee = landmarks[LEFT_KNEE_ID].y
            rknee = landmarks[RIGHT_KNEE_ID].y
            ##if lheel < lhips and rheel < rhips:
                ##return True
            if lheel < self.leftgroundheight and rheel < self.rightgroundheight:
                if lhips < self.prevlefthips and rhips < self.prevrighthips:
                    if lknee < self.prevleftknee and rknee < self.prevrightknee:
                        return True
                else:
                    self.leftgroundheight = lheel
                    self.rightgroundheight = rheel
                    self.prevlefthips = lhips
                    self.prevrighthips = rhips
                    self.prevleftknee = lknee
                    self.prevrightknee = rknee
        return False
            
