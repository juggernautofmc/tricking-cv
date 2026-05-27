import cv2
import mediapipe as mp

cap = cv2.VideoCapture("bomboclat.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break