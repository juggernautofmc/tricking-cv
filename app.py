import cv2

cap = cv2.VideoCapture("bomboclat.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break