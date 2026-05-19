import re
import os

# comment added on changes for testing purposes

path = r"c:\Users\yash jadhav\Desktop\wrapper_practice\Face_match\Face-match-test\frontend\src\FaceMatch.jsx"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Imports
if "import * as tf" not in content:
    content = content.replace('import * as faceapi from "face-api.js";', 'import * as faceapi from "face-api.js";\nimport * as tf from "@tensorflow/tfjs";\nimport * as cocoSsd from "@tensorflow-models/coco-ssd";')

# 2. Add new refs
if "cocoModelRef" not in content:
    content = content.replace('const isMeshDetectingRef = useRef(false);', 'const isMeshDetectingRef = useRef(false);\n  const cocoModelRef = useRef(null);\n  const gestureStepTimerRef = useRef(null);\n  const microMovementHistoryRef = useRef([]);')

# 3. Load COCO-SSD
if "cocoSsd.load(" not in content:
    content = content.replace("faceapi.nets.faceExpressionNet.loadFromUri(\"/models\"),", "faceapi.nets.faceExpressionNet.loadFromUri(\"/models\"),\n          cocoSsd.load().then(model => { cocoModelRef.current = model; }),")

# 4. Sustained frames to 3
content = content.replace("const SUSTAINED_FRAMES = 4;", "const SUSTAINED_FRAMES = 3;")

# 5. Overwrite gestures assignment in startCamera
old_gestures_block = """      const session_id = payload.session_id;
      const gestures = payload.gestures;
      if (!session_id || !Array.isArray(gestures) || gestures.length === 0) {"""
new_gestures_block = """      const session_id = payload.session_id || crypto.randomUUID();
      // RANDOMIZE 4 GESTURES as per requirement
      const allowed = ["turn_left", "turn_right", "move_closer", "smile"];
      const shuffled = allowed.sort(() => 0.5 - Math.random());
      const gestures = shuffled.slice(0, 4);
      if (!session_id || gestures.length !== 4) {"""
content = content.replace(old_gestures_block, new_gestures_block)

# 6. Black screen detection logic
old_black_screen = """      let blackPixels = 0;
      for (let i = 0; i < data.length; i += 4) {
        if (data[i] < 15 && data[i + 1] < 15 && data[i + 2] < 15) blackPixels++;
      }
      return blackPixels / (64 * 64) > 0.95; // >95% almost black"""
new_black_screen = """      let totalBrightness = 0;
      for (let i = 0; i < data.length; i += 4) {
        const brightness = (data[i] * 299 + data[i + 1] * 587 + data[i + 2] * 114) / 1000;
        totalBrightness += brightness;
      }
      return (totalBrightness / (64 * 64)) < 15; // Average brightness is very low"""
content = content.replace(old_black_screen, new_black_screen)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Applied initial set of modifications")
