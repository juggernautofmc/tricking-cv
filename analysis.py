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

LANDMARKS = [
    LEFT_SHOULDER_ID,
    RIGHT_SHOULDER_ID,
    LEFT_ELBOW_ID,
    RIGHT_ELBOW_ID,
    LEFT_WRIST_ID,
    RIGHT_WRIST_ID,
    LEFT_HIP_ID,
    RIGHT_HIP_ID,
    LEFT_KNEE_ID,
    RIGHT_KNEE_ID,
    LEFT_HEEL_ID,
    RIGHT_HEEL_ID
]

class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Analysis:
    def __init__(self):
        # input: none, output: none, useless function lol 
        self.leftgroundheight = None
        self.rightgroundheight = None

        self.prevhip = None #hip landmark of previous frame
        self.prevmidpthips = None #hip midpoint landmark of previous frame
        self.heel_vec = None

        self.frame_count = 0 #yup we tracking frames now
        self.stable_heel_frames = 0
        self.hip_rotation = 0 #trackin OBSERVABLE in-air hip rotation about the midpoint in degrees 
        # (note: this number is not accurate to 3d space)
        self.heel_distance = 0 #distance between the two heels
        self.l_knee_angle = 0 # angle in degrees @ left knee
        self.r_knee_angle = 0 # angle in degree @ right knee
        self.takeoff_foot = -1 # 0 (left foot), 1 (right foot), or 2 (both feet)
        self.landing_foot = -1 # 0 (left foot), 1 (right foot), or 2 (both feet)

        self.airborne = False
        self.inverted = False

    def compute(self, landmarks):

        lheel = landmarks[LEFT_HEEL_ID]
        rheel = landmarks[RIGHT_HEEL_ID]
        lknee = landmarks[LEFT_KNEE_ID]
        rknee = landmarks[RIGHT_KNEE_ID]
        lhips = landmarks[LEFT_HIP_ID]
        rhips = landmarks[RIGHT_HIP_ID]
            
        self.frame_count += 1

        if self.leftgroundheight is None:
            self.leftgroundheight = lheel.y
            self.rightgroundheight = rheel.y

        self.airborne = self.is_airborne(landmarks)
        self.heel_vec = (np.array([lheel.x, lheel.y]) - np.array([rheel.x, rheel.y]))
        self.heel_distance = np.linalg.norm(self.heel_vec)
        self.l_knee_angle = self.joint_angle(lhips, lknee, lheel)
        self.r_knee_angle = self.joint_angle(rhips, rknee, rheel)
        self.inverted = self.is_inverted(landmarks)

        if self.airborne:
            self.hip_rotation += self.calc_hip_rot(landmarks)
        else:
            self.hip_rotation = 0

        # input: 33 landmarks from pose.py, output: dict of metrics (joint angles, foot height, airborne)
        result = {
            "is_airborne": self.airborne,
            "is_inverted": self.inverted,
            "hip_rotation": self.hip_rotation,
            "heel_vec": self.heel_vec,
            "heel_dist": self.heel_distance,
            "l_knee_angle": self.l_knee_angle,
            "r_knee_angle": self.r_knee_angle,
            "frame_count": self.frame_count,
            "takeoff_foot": self.takeoff_foot,
            "landing_foot": self.landing_foot,
            "landmarks": self.export_landmarks(landmarks)
            }

        return result
    
    def export_landmarks(self, landmarks):
        exported = []

        for idx in LANDMARKS:
            lm = landmarks[idx]

            exported.append({
                "id": idx,
                "x": lm.x,
                "y": lm.y,
                "visibility": lm.visibility
            })
        
        return exported

    def calc_hip_rot(self, landmarks):
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
    # input: three landmarks (a, b, c) where b is the joint,
    # OR a and b are two NumPy vectors
    # output: UNSIGNED angle in degrees between the two vectors,
    # OR the angle at joint b

        p1 = np.array([a.x, a.y])
        vertex = np.array([b.x, b.y])
        p2 = np.array([c.x, c.y])

        vec1 = p1 - vertex
        vec2 = p2 - vertex

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0

        u1 = vec1 / norm1
        u2 = vec2 / norm2

        dot = np.dot(u1, u2)

        # numerical safety
        dot = np.clip(dot, -1.0, 1.0)

        return np.degrees(np.arccos(dot))
    
    def is_inverted(self, landmarks):
        shoulder_mid = (landmarks[LEFT_SHOULDER_ID].y + landmarks[RIGHT_SHOULDER_ID].y) / 2
        hips_mid = (landmarks[LEFT_HIP_ID].y + landmarks[RIGHT_HIP_ID].y) / 2

        return hips_mid > shoulder_mid
    
    def find_lower_foot(self, landmarks):
        lheel = landmarks[LEFT_HEEL_ID].y
        rheel = landmarks[RIGHT_HEEL_ID].y

        shoulder_mid = (landmarks[LEFT_SHOULDER_ID].y + landmarks[RIGHT_SHOULDER_ID].y) / 2
        heels_mid = (lheel + rheel) / 2

        body_height = abs(shoulder_mid - heels_mid)

        ONE_FOOT_THRESH = 0.04 * body_height

        diff = abs(lheel - rheel)

        if diff > ONE_FOOT_THRESH:
            if lheel > rheel:
                return 0 # == left foot
            else:
                return 1 # == right foot
        else:
            return 2 # == both feet

    def is_airborne(self, landmarks):
        # input: 33 landmarks, output: True if both feet are off the ground, False otherwise
            if landmarks is None:
                return False
            
            shoulder_mid = (landmarks[LEFT_SHOULDER_ID].y + landmarks[RIGHT_SHOULDER_ID].y) / 2
            heels_mid = (landmarks[LEFT_HEEL_ID].y + landmarks[RIGHT_HEEL_ID].y) / 2

            body_height = abs(shoulder_mid - heels_mid)

            TAKEOFF_THRESH = 0.040 * body_height
            LANDING_THRESH = 0.020 * body_height
            STABLE_THRESH = 0.250 * body_height

            lheel = landmarks[LEFT_HEEL_ID].y
            rheel = landmarks[RIGHT_HEEL_ID].y
            l_diff = self.leftgroundheight - lheel
            r_diff = self.rightgroundheight - rheel
            lowest = max(self.leftgroundheight, self.rightgroundheight)
            ld1 = lowest - rheel
            ld2 = lowest - lheel

            if not self.airborne:
                l_diff = self.leftgroundheight - lheel
                r_diff = self.rightgroundheight - rheel
                if l_diff > TAKEOFF_THRESH and r_diff > TAKEOFF_THRESH:
                    self.takeoff_foot = self.find_lower_foot(landmarks)
                    return True
                self.leftgroundheight = lheel
                self.rightgroundheight = rheel
                self.stable_heel_frames = 0
                return False
            else:
                l_diff = self.leftgroundheight - lheel
                r_diff = self.rightgroundheight - rheel
                if ld1 < LANDING_THRESH or ld2 < LANDING_THRESH:
                    self.leftgroundheight = lheel
                    self.rightgroundheight = rheel
                    self.stable_heel_frames = 0
                    self.landing_foot = self.find_lower_foot(landmarks)
                    return False
                elif ld1 < STABLE_THRESH or ld2 < STABLE_THRESH:
                    self.stable_heel_frames += 1
                    if self.stable_heel_frames > 10:
                        self.leftgroundheight = lheel
                        self.rightgroundheight = rheel
                        self.stable_heel_frames = 0
                        self.landing_foot = self.find_lower_foot(landmarks)
                        return False
                    return True
                else:
                    return True
            
