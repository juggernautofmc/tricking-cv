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
    # width = int(capt.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(capt.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    conf = ""

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # writer = cv2.VideoWriter(
    #         "output.mp4",
    #         fourcc,
    #         fps,
    #         (width, height)
    #     )

    in_air_frames = 0
    max_hip_rotation = 0
    front = 0
    side = 0
    heel_dist_onjump = 0
    
    while True:
        frame = capt.read()
        if frame is None:
            break
        

        landmarks = pos.process(frame, fps)

        if landmarks is not None: #gotta check if there are no people in frame so we dont pass none
            result = anly.compute(landmarks)
            airborne = result["is_airborne"]
            hip_rot = result["hip_rotation"]
            front = result["front_view_confidence"]
            side = result["side_view_confidence"]
            heel_dist = result["heel_dist"]

            if hip_rot > max_hip_rotation:
                max_hip_rotation = hip_rot

            if airborne:
                if in_air_frames == 2:
                    heel_dist_onjump = heel_dist
                in_air_frames += 1

            # ovly = overlay.Overlay(landmarks, width, height)
            # ovly.draw_overlay(frame, airborne, hip_rot)

            if front > side:
                conf = "Front View Confidence: " + str(front)
            else:
                conf = "Side View Confidence: " + str(side)
        
        # writer.write(frame) # make sure we writin the new frames and stuff
    print(conf)
    capt.release()
    # writer.release()
    pos.close()

    return {
        "in_air_frames": in_air_frames,
        "max_hip_rotation": max_hip_rotation,
        # "front_conf": front,
        # "side_conf": side,
        # these two metrics currently hurt model accuracy in their current implementation
        "heel_dist_onjump": float(heel_dist_onjump),
        "red_flag": ((in_air_frames < 8) or (in_air_frames > 35))
    }

if __name__ == "__main__":
    filepath = "input.mp4"
    process_video(filepath)
    

