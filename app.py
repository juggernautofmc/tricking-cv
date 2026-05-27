import cv2
import mediapipe as mp
import capture
import pose
import analysis
import overlay

# PURPOSE: main entry point — thin orchestrator, no logic lives here
# - open the video source via capture.py
# - loop over frames:
#     1. read a frame (capture.py)
#     2. detect landmarks (pose.py)
#     3. compute metrics (analysis.py)
#     4. draw and display (overlay.py)
# - stop the loop when the video ends or the user quits
