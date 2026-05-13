"""
HSV Tuner — run this to find correct HSV values for your camera/lighting
Shows live trackbars to dial in exact values.
Usage: python tune_hsv.py
Press S to save ranges, Q to quit.
"""
import cv2
import numpy as np

IMAGE_PATH = "test.jpg"
img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (900, int(900 * img.shape[0] / img.shape[1])))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def nothing(x): pass

cv2.namedWindow("Tuner")
for name, val in [("H_low",0),("H_high",179),("S_low",0),("S_high",255),("V_low",0),("V_high",255)]:
    cv2.createTrackbar(name, "Tuner", val, 179 if "H" in name else 255, nothing)

while True:
    hl = cv2.getTrackbarPos("H_low","Tuner")
    hh = cv2.getTrackbarPos("H_high","Tuner")
    sl = cv2.getTrackbarPos("S_low","Tuner")
    sh = cv2.getTrackbarPos("S_high","Tuner")
    vl = cv2.getTrackbarPos("V_low","Tuner")
    vh = cv2.getTrackbarPos("V_high","Tuner")

    mask   = cv2.inRange(hsv, np.array([hl,sl,vl]), np.array([hh,sh,vh]))
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("Tuner", result)
    cv2.imshow("Mask",  mask)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        print(f'lower=np.array([{hl},{sl},{vl}]), upper=np.array([{hh},{sh},{vh}])')
    if k == ord('q'):
        break

cv2.destroyAllWindows()