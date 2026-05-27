import cv2
import mediapipe as mp

# PURPOSE: draw everything onto the video frame for display
# - input: a raw frame + the metrics/landmarks computed by pose.py and analysis.py
# - draw the 33-point skeleton and joint connections over the athlete
# - draw a HUD showing useful numbers (joint angles, airborne state, etc.)
# - display the frame in a window and handle the quit key
# - this file should NOT do any math or detection — only drawing and display
