
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

def test_detection(image_path):
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        return

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1, # Very low for testing
        min_face_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    )
    
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read {image_path}")
            return
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        result = landmarker.detect(mp_image)
        if result.face_landmarks:
            print(f"Success! Detected {len(result.face_landmarks)} face(s) in {image_path}")
        else:
            print(f"Failed: No face detected in {image_path}")

if __name__ == "__main__":
    print("Testing elon.jpg...")
    test_detection("elon.jpg")
    print("\nTesting test.jpg...")
    test_detection("test.jpg")
