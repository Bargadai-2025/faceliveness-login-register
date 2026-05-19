"""
Compare face-match scores across preprocessing variants (local vs production drift).
Usage: python scratch/debug_match_pipeline.py [image_path]
"""
import asyncio
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from facenet_pytorch import MTCNN, InceptionResnetV1

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from database import close_db_pool, fetch, init_db_pool

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def embed_with_max_side(img_bgr: np.ndarray, max_side: int):
    h, w = img_bgr.shape[:2]
    img = img_bgr.copy()
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)
    if face is None:
        return None
    face = face.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(face).cpu().numpy()[0].astype("float32")
    return emb / (np.linalg.norm(emb) + 1e-6)


async def main():
    img_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(
            Path(__file__).resolve().parents[2]
            / ".cursor/projects/c-Users-yash-jadhav-Desktop-cursor-facematch-faceliveness/assets"
        )
    )
    # Default: workspace asset from user screenshot if present
    assets = list(
        (Path(__file__).resolve().parents[2].parent.parent / ".cursor" / "projects").glob(
            "**/WhatsApp_Image*.png"
        )
    )
    if len(sys.argv) <= 1 and assets:
        img_path = str(assets[0])

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        sys.exit(1)
    print(f"Image: {img_path} ({img.shape[1]}x{img.shape[0]})")
    print(f"Device: {DEVICE}\n")

    variants = {
        "index_build_db_800": 800,
        "legacy_match_1024": 1024,
        "register_no_resize": 99999,
    }

    await init_db_pool()
    try:
        rows = await fetch(
            "SELECT label, source, image_url, embedding FROM faces WHERE embedding IS NOT NULL"
        )
    finally:
        await close_db_pool()

    gallery = []
    for doc in rows:
        emb = doc.get("embedding")
        if not emb or len(emb) != 512:
            continue
        v = np.array(emb, dtype="float32")
        n = np.linalg.norm(v)
        if n == 0:
            continue
        gallery.append((doc["label"], doc["source"], v / n))

    print(f"Gallery size: {len(gallery)}\n")

    for name, max_side in variants.items():
        q = embed_with_max_side(img, max_side)
        if q is None:
            print(f"[{name}] NO FACE DETECTED\n")
            continue
        scores = [(lbl, src, float(np.dot(q, g))) for lbl, src, g in gallery]
        scores.sort(key=lambda x: x[2], reverse=True)
        print(f"=== {name} (max_side={max_side}) ===")
        for lbl, src, sc in scores[:8]:
            mark = " <-- Angel Mary" if lbl == "Angel Mary" else ""
            print(f"  {sc:.4f}  {lbl} ({src}){mark}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
