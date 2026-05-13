import cv2
import numpy as np
from face_detection import remove_background
import os

# Create a dummy image with a person-like shape
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img, (200, 100), (440, 400), (128, 128, 128), -1) # "Person"

# Try to remove background
processed = remove_background(img)

# If processed is different from img, it worked!
if not np.array_equal(img, processed):
    print("SUCCESS: Background was modified!")
    # Check if background became white (255, 255, 255)
    if np.any(processed[0, 0] == 255):
        print("CONFIRMED: Background is now white.")
else:
    print("FAILURE: Image was not changed.")
