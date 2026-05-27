import cv2
import mediapipe as mp

# PURPOSE: mediapipe wrapper
# - initialise the mediapipe pose detector
# - take a raw BGR frame as input, run pose detection on it
# - output a data object (you decide the shape) containing the 33 raw landmarks
#   each landmark has x, y, z (depth) and visibility (confidence) fields
# - return None (or similar) if no person is detected in the frame
#-Eshwar


class PoseDetector:

    def __init__(self):
        # input: none (optionally confidence thresholds if you want to configure them)
        # output: none, sets up the mediapipe pose object
        # used by: app.py once at startup before the frame loop begins

        self.mppose = mp.solutions.pose
        self.pose = self.mppose.Pose()
        

    def process(self, frame):
        # input: bgr frame from capture.py
        # output: returns the 33 landmarks for a pose/33 spots that mediapipe detects
        # app.py calls this process every frame and gives it to analysis.py
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #literally just swap to get rgb from bgr
        results = self.pose.process(rgb_frame)
        if results.pose_landmarks is None:
            return None
        return results.pose_landmarks

