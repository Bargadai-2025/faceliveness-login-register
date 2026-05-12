# 🛡️ FaceMatch & Liveness System Architecture

Welcome to the internal developer documentation. This guide explains how the entire biometric verification pipeline works, from the camera feed to the AI models and database matching.

---

## 🏗️ High-Level Overview

The system is split into three main parts:
1.  **Frontend (React)**: Captures camera frames and provides real-time UI feedback.
2.  **Backend (FastAPI)**: Orchestrates AI models and manages liveness "sessions."
3.  **Database (PostgreSQL + Cloudinary)**: Stores registered agent identities and their face "embeddings."

---

## 1️⃣ The Frontend Flow (`FaceMatch.jsx`)

The frontend is built to be "dumb but fast." It doesn't do heavy AI processing; it just handles the camera.

-   **Camera Loop**: Uses `requestAnimationFrame` to capture frames from the `<video>` element.
-   **Frame Streaming**: Converts frames to small JPEG blobs and sends them to the `/liveness/frame` endpoint every ~120ms.
-   **State Machine**:
    -   `calibration`: The user must stay still to set a "baseline."
    -   `gesture`: The user must perform randomized movements (Smile, Turn Left, etc.).
    -   `match`: Once liveness is verified, the system takes a high-quality selfie and sends it for database matching.

---

## 2️⃣ The Backend Brain (`api.py` & `frame_processor.py`)

The backend is where the magic happens. Every frame sent by the frontend goes through `process_frame`.

### 🧠 The AI Model Stack

| Model | Purpose | Why we use it? |
| :--- | :--- | :--- |
| **MTCNN / MediaPipe** | Face Detection | Detects 68 or 478 points on the face for precise tracking. |
| **FaceNet (InceptionResnetV1)** | Face Embeddings | Converts a face into a "DNA string" (512 numbers) for comparison. |
| **YOLOv8s** | Object Detection | Specifically trained to detect mobile phones and laptops to stop digital spoofs. |
| **FFT (Fast Fourier Transform)** | Frequency Analysis | Analyzes pixel flickering to detect if the user is looking at a digital screen. |

---

## 3️⃣ The Liveness Pipeline (Anti-Spoofing)

We use a "Defense in Depth" strategy. Multiple layers of security run simultaneously:

### 📍 Layer 1: Passive Liveness (Static)
-   **Texture Analysis**: Looks for the "moiré pattern" (grainy pixels) typical of screens.
-   **ROI Luminance**: Real skin reflects light; screens emit it. We check the distribution of brightness.

### 📍 Layer 2: 3D Depth (Parallax)
-   **Displacement Check**: In a real 3D face, the nose moves more than the ears when you turn. On a flat photo or screen, everything moves together. We measure this "relative movement" ratio.

### 📍 Layer 3: Micro-Expressions
-   Detects tiny, involuntary jitters in the eyes and lips. Replay videos or photos are "too static" and fail this check.

### 📍 Layer 4: Interactive Gestures
-   The system asks for random movements (e.g., "SMILE").
-   **Snappy Detection**: We use `SUSTAINED_FRAMES` and optimized thresholds in `liveness_checks.py` to recognize these gestures instantly.

---

## 4️⃣ Face Matching & Database Selection

Once the user passes liveness, the `/match` endpoint is called.

### 🧬 What is an "Embedding"?
Instead of comparing raw pixels (which change with lighting), we use **FaceNet** to generate a 512-dimensional vector. 
- Two photos of the same person will have vectors that are very "close" in mathematical space.

### 🔎 Comparison Logic (Vector Search)
1.  **Fetch All**: We load all registered agent embeddings from PostgreSQL.
2.  **Cosine Similarity**: We calculate the "dot product" between the captured selfie and every image in the DB.
3.  **Confidence Score**: 
    -   `1.0` = Perfect match.
    -   `< 0.8` = Suspicious.
4.  **Security Penalty**: If the liveness check detected suspicious activity (like a phone in the background), we deduct points from the final match score (e.g., -30%).

---

## 📁 Key Files Summary

-   **`api.py`**: The FastAPI server. Handles registration and matching.
-   **`frame_processor.py`**: The logic for the real-time liveness loop.
-   **`liveness_checks.py`**: Mathematical thresholds for depth, gestures, and texture.
-   **`face_detection.py`**: Low-level AI utilities (MTCNN, Ear/Mar calculation).
-   **`FaceMatch.jsx`**: The React component for the user dashboard.
-   **`loginpage.jsx`**: Handles identity selection from the DB.

---

## 🚀 Junior Dev Tips
1.  **Don't block the loop**: `process_frame` must run fast (~50-100ms). Never do slow DB calls inside it.
2.  **Indentation is Life**: In `api.py`, be very careful with whitespace. One wrong space can break the whole function.
3.  **Thresholds**: If liveness is too hard, adjust `SUSTAINED_FRAMES` or `_threshold` fractions in `liveness_checks.py`.

---
*Document Version 1.2 — 2026*