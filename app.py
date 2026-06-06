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
# - Sami and Eshwar


def main(source):
    # input: filepath 4 video, output: none
    # opens video with capture.py, then loops through calling the other files until we done
    capt = capture.VideoCapture(source)
    pos = pose.PoseDetector()
    anly = analysis.Analysis()

    fps = capt.cap.get(cv2.CAP_PROP_FPS)
    width = int(capt.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capt.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
            "goon.mp4",
            fourcc,
            fps,
            (width, height)
        )

    while True:
        frame = capt.read()
        if frame is None:
            break
        

        landmarks = pos.process(frame, fps)

        if landmarks is not None: #gotta check if there are no people in frame so we dont pass none
            ##metrics = anly.compute(landmarks)
            anly.compute(landmarks)
            airborne = anly.is_airborne(landmarks)

            ovly = overlay.Overlay(landmarks, width, height)
            ovly.draw_overlay(frame, airborne)
        
        writer.write(frame) # make sure we writin the new frames and stuff
    capt.release()
    writer.release()
    pos.close()

if __name__ == "__main__":
    filepath = "backflippinggoonsesh.mp4"
    main(filepath)
    

