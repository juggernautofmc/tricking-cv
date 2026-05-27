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


def main(source):
    # input: filepath 4 video, output: none
    # opens video with capture.py, then loops through calling the other files until we done
    cap = capture.VideoCapture(source)
    pos = pose.PoseDetector()
    anly = analysis.Analysis()

    while True:
        frame = cap.read()
        if frame is None:
            break
        
        landmarks = pos.process(frame)
        if landmarks is None: #gotta check if there are no people in frame so we dont pass none
            overlay.draw(frame, None, None) 
        else:
            metrics = anly.compute(landmarks)
            overlay.draw(frame, landmarks, metrics)
    cap.release()
filepath = "backflippinggoonsesh.mp4"
main(filepath)
    

