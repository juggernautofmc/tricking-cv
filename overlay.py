import cv2
import mediapipe as mp

# PURPOSE: draw everything onto the video frame for display
# - input: a raw frame + the metrics/landmarks computed by pose.py and analysis.py
# - draw the 33-point skeleton and joint connections over the athlete
# - draw a HUD showing useful numbers (joint angles, airborne state, etc.)
# - display the frame in a window and handle the quit key
# - this file should NOT do any math or detection — only drawing and display

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

def overlay(frame, pose):
    pts = pose.pose_landmarks[0]

    ## LEFT ARM DRAWING
    l_shoulder_x = pts[LEFT_SHOULDER_ID].x
    l_shoulder_y = pts[LEFT_SHOULDER_ID].y
    l_elbow_x = pts[LEFT_ELBOW_ID].x
    l_elbow_y = pts[LEFT_ELBOW_ID].y
    l_wrist_x = pts[LEFT_WRIST_ID].x
    l_wrist_y = pts[LEFT_WRIST_ID].y
    