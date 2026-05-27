import cv2

# PURPOSE: handle all video input
# - open a video source — either a file path or a webcam index (0, 1, ...)
# - read frames one at a time and return them to the main loop
# - cleanly release the video source and close windows when done
# - this file should NOT do any pose detection or drawing
# - Eshwar


class VideoCapture:

    def __init__(self, source):
        # input: source should be like a file path to example.mp4
        # output: none lol, bust initialize self.cap
        self.cap = cv2.VideoCapture(source)

        


    def read(self):
        # input: none
        # output: frame (numpy array) if successful, None if video is done or failed
        pass

    def release(self):
        # input: none
        # output: none, releases the video source and closes windows
        pass

