import mediapipe as mp

# PURPOSE: mediapipe wrapper
# - initialise the mediapipe pose detector
# - take a raw BGR frame as input, run pose detection on it
# - output a data object (you decide the shape) containing the 33 raw landmarks
#   each landmark has x, y, z (depth) and visibility (confidence) fields
# - return None (or similar) if no person is detected in the frame
