import cv2
import numpy as np
from face_detection import remove_background
import os

# Use a real image from the project
if os.path.exists("test.jpg"):
    img = cv2.imread("test.jpg")
    processed = remove_background(img)
    if not np.array_equal(img, processed):
        print("SUCCESS: Real image background was modified!")
        cv2.imwrite("test_processed.jpg", processed)
        print("Processed image saved as test_processed.jpg")
    else:
        print("FAILURE: Real image was not changed. Segmenter might not see a person.")
else:
    print("test.jpg not found")
