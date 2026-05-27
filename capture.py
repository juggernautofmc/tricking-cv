import cv2

# PURPOSE: handle all video input
# - open a video source — either a file path or a webcam index (0, 1, ...)
# - read frames one at a time and return them to the main loop
# - cleanly release the video source and close windows when done
# - this file should NOT do any pose detection or drawing
