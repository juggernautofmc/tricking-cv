import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# PURPOSE: mediapipe wrapper
# - initialise the mediapipe pose detector
# - take a raw BGR frame as input, run pose detection on it
# - output a data object (you decide the shape) containing the 33 raw landmarks
#   each landmark has x, y, z (depth) and visibility (confidence) fields
# - return None (or similar) if no person is detected in the frame
#-Sami and Eshwar


class PoseDetector:

    def __init__(self):
        # input: none (optionally confidence thresholds if you want to configure them)
        # output: none, sets up the mediapipe pose object
        # used by: app.py once at startup before the frame loop begins

        base_options = python.BaseOptions(
            model_asset_path="pose_landmarker_full.task"
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.frame_index = 0
        

    def process(self, frame, fps):
        # input: bgr frame from capture.py
        # output: returns the 33 landmarks for a pose/33 spots that mediapipe detects
        # app.py calls this process every frame and gives it to analysis.py
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #literally just swap to get rgb from bgr
        
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        ) ## MediaPipe doesn't accept a numpy frame, create mediapipe Image object with the frame data

        timestamp_ms = int((self.frame_index / fps) * 1000) ## calculate timestamp in ms

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        self.frame_index += 1
        ## go to next frame

        if not result.pose_landmarks:
            return None

        return result.pose_landmarks[0]
    
    def close(self): ##closes landmarker
        self.landmarker.close()
