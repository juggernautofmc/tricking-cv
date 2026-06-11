import cv2

# PURPOSE: handle all video input
# - open a video source — either a file path or a webcam index (0, 1, ...)
# - read frames one at a time and return them to the main loop
# - cleanly release the video source and close windows when done
# - this file should NOT do any pose detection or drawing
# - Eshwar


class VideoCapture:

    def __init__(self, source, target_fps=None):
        # input: source should be like a file path to example.mp4
        # output: none lol, bust initialize self.cap
        self.cap = cv2.VideoCapture(source)
        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.target_fps = target_fps

        self.target_dt_ms = None
        self.next_time_ms = 0

        if target_fps is not None:
            self.target_dt_ms = 1000 / target_fps

    def read(self):
        # input: none
        # output: frame (numpy array) if successful, None if video is done or failed
        while True:
            success, frame = self.cap.read()
            if not success:
                return None
            
            if self.target_fps is None:
                return frame
            
            curr_time_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)

            if curr_time_ms >= self.next_time_ms:
                self.next_time_ms += self.target_dt_ms
                return frame

    def release(self):
        # input: none
        # output: none, releases the video source and closes windows
        self.cap.release()
        cv2.destroyAllWindows()
