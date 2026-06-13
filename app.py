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


def process_video(source):
    # input: filepath 4 video, output: none
    # opens video with capture.py, then loops through calling the other files until we done
    capt = capture.VideoCapture(source, target_fps=30)
    pos = pose.PoseDetector()
    anly = analysis.Analysis()

    fps = 30
    width = int(capt.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capt.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
            "output.mp4",
            fourcc,
            fps,
            (width, height)
        )

    in_air_frames = 0
    max_hip_rotation = 0
    landing = None
    takeoff = None
    
    while True:
        frame = capt.read()
        if frame is None:
            break
        

        landmarks = pos.process(frame, fps)

        if landmarks is not None: #gotta check if there are no people in frame so we dont pass none
            result = anly.compute(landmarks)
            airborne = result["is_airborne"]
            hip_rot = result["hip_rotation"]
            takeoff = result["takeoff_foot"]
            landing = result["landing_foot"]

            if hip_rot > max_hip_rotation:
                max_hip_rotation = hip_rot

            if airborne:
                in_air_frames += 1

            ovly = overlay.Overlay(landmarks, width, height)
            ovly.draw_overlay(frame, airborne, hip_rot)
        
        writer.write(frame) # make sure we writin the new frames and stuff
    capt.release()
    writer.release()
    pos.close()

    return {
        "in_air_frames": in_air_frames,
        "max_hip_rotation": max_hip_rotation,
        "takeoff_foot": int(takeoff),
        "landing_foot": int(landing),
        "red_flag": ((in_air_frames < 8) or (in_air_frames > 35))
    }

if __name__ == "__main__":
    filepath = "dataset_videos/540-kick/IMG_2560 4.MOV"
    result = process_video(filepath)
    print("takeoff foot:", result["takeoff_foot"])
    print("landing foot:", result["landing_foot"])
    

