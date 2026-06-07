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

class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Analysis:
    def __init__(self):
        # input: none, output: none, useless function lol 
        self.leftgroundheight = None
        self.rightgroundheight = None
        self.startlefthip = None
        self.startrighthip = None
        self.startleftknee = None
        self.startrightknee = None
        self.prevhip = None
        self.prevmidpthips = None
        self.frame_count = 0 #yup we tracking frames now
        self.in_air_rotation = 0

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
                self.startlefthip = lhips
                self.startrighthip = rhips
                self.startleftknee = lknee
                self.startrightknee = rknee
            #else:
                #self.groundheight = min(self.groundheight, lheel, rheel)
        airborne = self.is_airborne(landmarks)
        if airborne:
            self.in_air_rotation += self.calc_rotation(landmarks)
        else:
            self.in_air_rotation = 0

        # input: 33 landmarks from pose.py, output: dict of metrics (joint angles, foot height, airborne)
        result = {"is_airborne": airborne,
                  "in_air_rotation": self.in_air_rotation}

        return result
    
    def calc_rotation(self, landmarks):
        curr_midpt_x = (landmarks[LEFT_HIP_ID].x + landmarks[RIGHT_HIP_ID].x) / 2
        curr_midpt_y = (landmarks[LEFT_HIP_ID].y + landmarks[RIGHT_HIP_ID].y) / 2
        curr_midpt = MockLandmark(curr_midpt_x, curr_midpt_y)

        if self.prevmidpthips is None or self.prevhip is None:
            self.prevmidpthips = curr_midpt
            self.prevhip = landmarks[LEFT_HIP_ID]
            return 0

        dx_midpt = curr_midpt.x - self.prevmidpthips.x
        dy_midpt = curr_midpt.y - self.prevmidpthips.y

        adjusted_hip_x = self.prevhip.x + dx_midpt
        adjusted_hip_y = self.prevhip.y + dy_midpt
        init_hip = MockLandmark(adjusted_hip_x, adjusted_hip_y)

        curr_hip = landmarks[LEFT_HIP_ID]

        self.prevhip = curr_hip
        self.prevmidpthips = curr_midpt

        return self.joint_angle(curr_hip, curr_midpt, init_hip)

    def joint_angle(self, a, b, c):
        # input: three landmarks (a, b, c) where b is the joint, output: angle at b in degrees
        p1 = np.array([a.x, a.y])
        vertex = np.array([b.x, b.y])
        p2 = np.array([c.x, c.y])

        vec1 = p1 - vertex
        vec2 = p2 - vertex

        dot = np.dot(vec1, vec2)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return np.degrees(np.arccos(dot / (norm1 * norm2)))

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
                if lhips < self.startlefthip and rhips < self.startrighthip:
                    if lknee < self.startleftknee and rknee < self.startrightknee:
                        return True
                else:
                    self.leftgroundheight = lheel
                    self.rightgroundheight = rheel
                    self.startlefthip = lhips
                    self.startrighthip = rhips
                    self.startleftknee = lknee
                    self.startrightknee = rknee
        return False
            
