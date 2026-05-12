import os

path = r"c:\Users\yash jadhav\Desktop\wrapper_practice\Face_match\Face-match-test\frontend\src\FaceMatch.jsx"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update runLivenessLoop to use detectAllFaces
old_detect = """    try {
      const detection = await faceapi
        .detectSingleFace(
          video,
          new faceapi.TinyFaceDetectorOptions({
            inputSize: LIVENESS_FACE_INPUT_SIZE,
            scoreThreshold: LIVENESS_SCORE_THRESHOLD,
          }),
        )
        .withFaceLandmarks()
        .withFaceExpressions();

      if (detection) {"""

new_detect = """    try {
      if (cocoModelRef.current) {
        const preds = await cocoModelRef.current.detect(video);
        const spoofObjs = ["cell phone", "laptop", "tablet", "monitor", "tv"];
        const hasSpoof = preds.some(p => spoofObjs.includes(p.class));
        if (hasSpoof) {
          setError("Mobile or electronic device detected. Verification stopped");
          stopMediaOnly();
          setTimeout(() => { startCamera(); }, 2000);
          return;
        }
      }

      const detections = await faceapi
        .detectAllFaces(
          video,
          new faceapi.TinyFaceDetectorOptions({
            inputSize: LIVENESS_FACE_INPUT_SIZE,
            scoreThreshold: 0.6,
          }),
        )
        .withFaceLandmarks()
        .withFaceExpressions();

      if (detections && detections.length > 1) {
        setError("Multiple faces detected! Please ensure only one person is in the frame.");
        stopMediaOnly();
        setTimeout(() => { startCamera(); }, 2000);
        return;
      }

      if (detections && detections.length === 1) {
        const detection = detections[0];
        const pts = detection.landmarks.positions;
        const faceW = Math.abs(pts[16].x - pts[0].x);
        
        if (faceW < video.videoWidth * 0.15) {
          faceLostCountRef.current += 1;
          // Too far, skip processing
          isDetectingRef.current = false;
          return;
        }
        
        // Micro-movement detection
        if (!baselineRef.current) {
            microMovementHistoryRef.current.push(pts[30]);
            if (microMovementHistoryRef.current.length > 20) {
                microMovementHistoryRef.current.shift();
            }
        } else {
            microMovementHistoryRef.current.push(pts[30]);
            if (microMovementHistoryRef.current.length > 20) {
                microMovementHistoryRef.current.shift();
                // Check if completely static
                const xs = microMovementHistoryRef.current.map(p => p.x);
                const ys = microMovementHistoryRef.current.map(p => p.y);
                const dx = Math.max(...xs) - Math.min(...xs);
                const dy = Math.max(...ys) - Math.min(...ys);
                if (dx < 0.5 && dy < 0.5) {
                    setError("Static replay attempt detected. Restarting.");
                    stopMediaOnly();
                    setTimeout(() => { startCamera(); }, 2000);
                    return;
                }
            }
        }
"""
content = content.replace(old_detect, new_detect)

# 2. Add Timer logic for 5-second timeout
# We will intercept `setChallengeIndex` to start timeout, and `resetLiveness` to clear it.
old_mark_done = """        challengeIndexRef.current = nextIndex;
        setChallengeIndex(nextIndex);
        setChallengeMsg("");
"""
new_mark_done = """        challengeIndexRef.current = nextIndex;
        setChallengeIndex(nextIndex);
        setChallengeMsg("");
        
        if (gestureStepTimerRef.current) clearTimeout(gestureStepTimerRef.current);
        gestureStepTimerRef.current = setTimeout(() => {
            setError("Face verification timeout. Restarting");
            stopMediaOnly();
            setTimeout(() => { startCamera(); }, 2000);
        }, 5000);
"""
content = content.replace(old_mark_done, new_mark_done)

old_reset_liveness = """    if (gestureTimerRef.current) {
      clearTimeout(gestureTimerRef.current);
      gestureTimerRef.current = null;
    }"""
new_reset_liveness = """    if (gestureTimerRef.current) {
      clearTimeout(gestureTimerRef.current);
      gestureTimerRef.current = null;
    }
    if (gestureStepTimerRef.current) {
      clearTimeout(gestureStepTimerRef.current);
      gestureStepTimerRef.current = null;
    }
    microMovementHistoryRef.current = [];
"""
content = content.replace(old_reset_liveness, new_reset_liveness)

# Stop camera addition
old_stop_camera = """    if (gestureTimerRef.current) {
      clearTimeout(gestureTimerRef.current);
      gestureTimerRef.current = null;
    }
    setShowCamera(false);"""
new_stop_camera = """    if (gestureTimerRef.current) {
      clearTimeout(gestureTimerRef.current);
      gestureTimerRef.current = null;
    }
    if (gestureStepTimerRef.current) {
      clearTimeout(gestureStepTimerRef.current);
      gestureStepTimerRef.current = null;
    }
    setShowCamera(false);"""
content = content.replace(old_stop_camera, new_stop_camera)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Applied second set of modifications")
