import cv2
import mediapipe as mp

# PURPOSE: draw everything onto the video frame for display
# - input: a raw frame + the metrics/landmarks computed by pose.py and analysis.py
# - draw the 33-point skeleton and joint connections over the athlete
# - draw a HUD showing useful numbers (joint angles, airborne state, etc.)
# - display the frame in a window and handle the quit key
# - this file should NOT do any math or detection — only drawing and display
# - Sami

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

## COLORS (B,G,R)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
PINK = (193, 182, 255)



class Overlay:
    
    def __init__(self, landmarks, width, height):

        ## LEFT ARM
        self.lx_shoulder = int(landmarks[LEFT_SHOULDER_ID].x * width)
        self.ly_shoulder = int(landmarks[LEFT_SHOULDER_ID].y * height)
        self.lv_shoulder = landmarks[LEFT_SHOULDER_ID].visibility

        self.lx_elbow = int(landmarks[LEFT_ELBOW_ID].x * width)
        self.ly_elbow = int(landmarks[LEFT_ELBOW_ID].y * height)
        self.lv_elbow = landmarks[LEFT_ELBOW_ID].visibility

        self.lx_wrist = int(landmarks[LEFT_WRIST_ID].x * width)
        self.ly_wrist = int(landmarks[LEFT_WRIST_ID].y * height)
        self.lv_wrist = landmarks[LEFT_WRIST_ID].visibility

        ## RIGHT ARM
        self.rx_shoulder = int(landmarks[RIGHT_SHOULDER_ID].x * width)
        self.ry_shoulder = int(landmarks[RIGHT_SHOULDER_ID].y * height)
        self.rv_shoulder = landmarks[RIGHT_SHOULDER_ID].visibility

        self.rx_elbow = int(landmarks[RIGHT_ELBOW_ID].x * width)
        self.ry_elbow = int(landmarks[RIGHT_ELBOW_ID].y * height)
        self.rv_elbow = landmarks[RIGHT_ELBOW_ID].visibility

        self.rx_wrist = int(landmarks[RIGHT_WRIST_ID].x * width)
        self.ry_wrist = int(landmarks[RIGHT_WRIST_ID].y * height)
        self.rv_wrist = landmarks[RIGHT_WRIST_ID].visibility

        ## LEFT LEG
        self.lx_hip = int(landmarks[LEFT_HIP_ID].x * width)
        self.ly_hip = int(landmarks[LEFT_HIP_ID].y * height)
        self.lv_hip = landmarks[LEFT_HIP_ID].visibility

        self.lx_knee = int(landmarks[LEFT_KNEE_ID].x * width)
        self.ly_knee = int(landmarks[LEFT_KNEE_ID].y * height)
        self.lv_knee = landmarks[LEFT_KNEE_ID].visibility

        self.lx_heel = int(landmarks[LEFT_HEEL_ID].x * width)
        self.ly_heel = int(landmarks[LEFT_HEEL_ID].y * height)
        self.lv_heel = landmarks[LEFT_HEEL_ID].visibility

        ## RIGHT LEG
        self.rx_hip = int(landmarks[RIGHT_HIP_ID].x * width)
        self.ry_hip = int(landmarks[RIGHT_HIP_ID].y * height)
        self.rv_hip = landmarks[RIGHT_HIP_ID].visibility

        self.rx_knee = int(landmarks[RIGHT_KNEE_ID].x * width)
        self.ry_knee = int(landmarks[RIGHT_KNEE_ID].y * height)
        self.rv_knee = landmarks[RIGHT_KNEE_ID].visibility

        self.rx_heel = int(landmarks[RIGHT_HEEL_ID].x * width)
        self.ry_heel = int(landmarks[RIGHT_HEEL_ID].y * height)
        self.rv_heel = landmarks[RIGHT_HEEL_ID].visibility

    def draw_point(self, frame, x, y, color):
        cv2.circle(frame, (x, y), 10, color, -1)

    def draw_line(self, frame, pt1, pt2, color):
        cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
    
    def draw_overlay(self, frame):
        # LEFT ARM
        if self.lv_shoulder > 0.5:
            self.draw_point(frame, self.lx_shoulder, self.ly_shoulder, RED)
        if self.lv_elbow > 0.5:
            if self.lv_shoulder > 0.5:
                self.draw_line(frame, (self.lx_shoulder, self.ly_shoulder), (self.lx_elbow, self.ly_elbow), RED)
            self.draw_point(frame, self.lx_elbow, self.ly_elbow, RED)
        if self.lv_wrist > 0.5:
            if self.lv_elbow > 0.5:
                self.draw_line(frame, (self.lx_elbow, self.ly_elbow), (self.lx_wrist, self.ly_wrist), RED)
            self.draw_point(frame, self.lx_wrist, self.ly_wrist, RED)
        
        # RIGHT ARM
        if self.rv_shoulder > 0.5:
            self.draw_point(frame, self.rx_shoulder, self.ry_shoulder, GREEN)
        if self.rv_elbow > 0.5:
            if self.rv_shoulder > 0.5:
                self.draw_line(frame, (self.rx_shoulder, self.ry_shoulder), (self.rx_elbow, self.ry_elbow), GREEN)
            self.draw_point(frame, self.rx_elbow, self.ry_elbow, GREEN)
        if self.rv_wrist > 0.5:
            if self.rv_elbow > 0.5:
                self.draw_line(frame, (self.rx_elbow, self.ry_elbow), (self.rx_wrist, self.ry_wrist), GREEN)
            self.draw_point(frame, self.rx_wrist, self.ry_wrist, GREEN)

        # LEFT LEG
        if self.lv_hip > 0.5:
            self.draw_point(frame, self.lx_hip, self.ly_hip, BLUE)
        if self.lv_knee > 0.5:
            if self.lv_hip > 0.5:
                self.draw_line(frame, (self.lx_hip, self.ly_hip), (self.lx_knee, self.ly_knee), BLUE)
            self.draw_point(frame, self.lx_knee, self.ly_knee, BLUE)
        if self.lv_heel > 0.5:
            if self.lv_knee > 0.5:
                self.draw_line(frame, (self.lx_knee, self.ly_knee), (self.lx_heel, self.ly_heel), BLUE)
            self.draw_point(frame, self.lx_heel, self.ly_heel, BLUE)

        # RIGHT LEG
        if self.rv_hip > 0.5:
            self.draw_point(frame, self.rx_hip, self.ry_hip, YELLOW)
        if self.rv_knee > 0.5:
            if self.rv_hip > 0.5:
                self.draw_line(frame, (self.rx_hip, self.ry_hip), (self.rx_knee, self.ry_knee), YELLOW)
            self.draw_point(frame, self.rx_knee, self.ry_knee, YELLOW)
        if self.rv_heel > 0.5:
            if self.rv_knee > 0.5:
                self.draw_line(frame, (self.rx_knee, self.ry_knee), (self.rx_heel, self.ry_heel), YELLOW)
            self.draw_point(frame, self.rx_heel, self.ry_heel, YELLOW)
        
        # MIDPOINTS
        hips_midpt_x = (self.rx_hip + self.lx_hip) // 2
        hips_midpt_y = (self.ry_hip + self.ly_hip) // 2
        shoulders_midpt_x = (self.rx_shoulder + self.lx_shoulder) // 2
        shoulders_midpt_y = (self.ry_shoulder + self.ly_shoulder) // 2

        if self.rv_shoulder > 0.5 and self.lv_shoulder > 0.5:
            self.draw_point(frame, shoulders_midpt_x, shoulders_midpt_y, PINK)
            self.draw_line(frame, (self.rx_shoulder, self.ry_shoulder), (shoulders_midpt_x, shoulders_midpt_y), PINK)
            self.draw_line(frame, (self.lx_shoulder, self.ly_shoulder), (shoulders_midpt_x, shoulders_midpt_y), PINK)
            if self.rv_hip > 0.5 and self.lv_hip > 0.5:
                self.draw_line(frame, (shoulders_midpt_x, shoulders_midpt_y), (hips_midpt_x, hips_midpt_y), PINK)
        
        if self.rv_hip > 0.5 and self.lv_hip > 0.5:
            self.draw_point(frame, hips_midpt_x, hips_midpt_y, PINK)
            self.draw_line(frame, (self.rx_hip, self.ry_hip), (hips_midpt_x, hips_midpt_y), PINK)
            self.draw_line(frame, (self.lx_hip, self.ly_hip), (hips_midpt_x, hips_midpt_y), PINK)
        
