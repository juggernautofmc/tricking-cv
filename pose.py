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
        # input: frame — a BGR numpy array from capture.py
        # output: pose_landmarks object (33 landmarks) if a person is detected, None otherwise
        # used by: app.py on every frame, result is passed to analysis.py
        pass
