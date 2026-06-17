from ultralytics import YOLO

# PURPOSE: YOLO pose wrapper
# - initialise the YOLO pose detector
# - take a raw BGR frame as input, run pose detection on it
# - output the 17 YOLO/COCO pose landmarks
# - each landmark has x, y, z, and visibility fields
# - return None if no person is detected in the frame


class Landmark:
    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class PoseDetector:
    def __init__(self, model_path="yolo26l-pose.pt", conf=0.25, imgsz=640):
        # If yolo26n-pose.pt is not available in your local Ultralytics install,
        # use another YOLO pose model such as yolo11n-pose.pt.
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def process(self, frame, fps=None):
        # input: BGR frame from capture.py
        # output: list of 17 YOLO/COCO pose landmarks
        results = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )

        if not results:
            return None

        result = results[0]

        if result.keypoints is None or result.keypoints.xyn is None:
            return None

        xyn = result.keypoints.xyn
        if len(xyn) == 0:
            return None

        person_idx = self._best_person_index(result)
        keypoints_xy = xyn[person_idx].cpu().numpy()

        if result.keypoints.conf is not None:
            keypoints_conf = result.keypoints.conf[person_idx].cpu().numpy()
        else:
            keypoints_conf = [1.0] * len(keypoints_xy)

        landmarks = []
        for point, confidence in zip(keypoints_xy, keypoints_conf):
            landmarks.append(
                Landmark(
                    x=float(point[0]),
                    y=float(point[1]),
                    z=0.0,
                    visibility=float(confidence),
                )
            )

        return landmarks

    def close(self):
        # Kept so app.py can call pos.close() for either backend.
        pass

    def _best_person_index(self, result):
        if result.boxes is None or result.boxes.xyxy is None or len(result.boxes.xyxy) == 0:
            return 0

        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return int(areas.argmax())
