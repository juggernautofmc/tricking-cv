import cv2
# import os
import capture
import pose
import analysis
import overlay
import classifier

# PURPOSE: main entry point — thin orchestrator, no logic lives here
# - open the video source via capture.py
# - loop over frames:
#     1. read a frame (capture.py)
#     2. detect landmarks (pose.py)
#     3. compute metrics (analysis.py)
#     4. draw and display (overlay.py)
# - stop the loop when the video ends or the user quits
# - Sami and Eshwar

def process_video(source, output_name):
    # input: filepath 4 video, output: none
    # opens video with capture.py, then loops through calling the other files until we done
    capt = capture.VideoCapture(source, target_fps=30)
    pos = pose.PoseDetector()

    fps = 30
    width = int(capt.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capt.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    anly = analysis.Analysis(width, height)

    # os.makedirs("output_videos", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
            output_name,
            fourcc,
            fps,
            (width, height)
        )

    in_air_frames = 0
    max_hip_rotation = 0
    landing = -1
    takeoff = -1
    l_ft_landing = 0
    
    while True:
        frame = capt.read()
        if frame is None:
            break

        airborne = False
        landmarks = pos.process(frame, fps)

        if landmarks is not None: #gotta check if there are no people in frame so we dont pass none
            result = anly.compute(landmarks)
            airborne = result["is_airborne"]
            hip_rot = result["hip_rotation"]
            takeoff = result["takeoff_foot"]
            landing = result["landing_foot"]
            l_ft_landing = result["left_ft_ht"]

            if hip_rot > max_hip_rotation:
                max_hip_rotation = hip_rot

            if airborne:
                in_air_frames += 1


            ovly = overlay.Overlay(landmarks, width, height)
            ovly.draw_overlay(frame, airborne, hip_rot)
        else:
            airborne = anly.handle_missing_detection()
        
        writer.write(frame) # make sure we writin the new frames and stuff
        if not airborne and in_air_frames > 8 and max_hip_rotation > 30:
            break

    capt.release()
    writer.release()
    pos.close()

    return {
        "in_air_frames": in_air_frames,
        "max_hip_rotation": max_hip_rotation,
        "takeoff_foot": int(takeoff),
        "landing_foot": int(landing),
        "left_ft_ht": l_ft_landing,
        "red_flag": ((in_air_frames < 8) or (in_air_frames > 35) or max_hip_rotation > 600)
    }

if __name__ == "__main__":
    filepath = "input3.MOV"
    model = classifier.TrickClassifier()
    result = process_video(filepath, "output3.mp4")
    # if result["red_flag"]:
    #     print("Pose extraction failed quality checks.")
    # else:
    prediction = model.predict(result)

    print("Prediction:", prediction["prediction"])
    print("Confidence:", prediction["confidence"])
    print("All probabilities:", prediction["probabilities"])
    
