import mediapipe as mp
try:
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    print("SUCCESS: SelfieSegmentation initialized")
except Exception as e:
    print(f"FAILURE: {e}")
