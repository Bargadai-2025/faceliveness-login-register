# build_db.py
import os
import cv2
import torch
import numpy as np
import cloudinary
import cloudinary.uploader
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
from dotenv import load_dotenv
from database import execute_sync, insert_face_sync

load_dotenv()

# Cloudinary setup
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Tracking files: record which photos are already indexed (so we can append only new ones)
INDEXED_BARGAD_FILE = "indexed_bargad.txt"
INDEXED_LFW_FILE = "indexed_lfw.txt"


def load_tracked(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def append_tracked(path, key):
    with open(path, "a") as f:
        f.write(key + "\n")


# ---- Prompt ----
confirm_drop = input("🗑️ Drop existing data and rebuild everything? (y/n): ")
if confirm_drop.lower() == "y":
    execute_sync("DELETE FROM faces")
    for f in (INDEXED_BARGAD_FILE, INDEXED_LFW_FILE):
        if os.path.exists(f):
            os.remove(f)
    print("🗑️ Cleared existing PostgreSQL rows and tracking files\n")
    index_bargad = True
    append_only = False
else:
    append_choice = input("📥 Append only NEW photos (Bargad + LFW), skip already indexed? (y/n): ")
    if append_choice.lower() == "y":
        append_only = True
        index_bargad = True  # we will process bargad in append-only mode
        if not os.path.exists(INDEXED_BARGAD_FILE) and not os.path.exists(INDEXED_LFW_FILE):
            print("⚠️ No tracking files yet. Run a full rebuild (y) once to create them, then use append-only for new photos.\n")
        print("✅ Append-only mode: only new photos will be processed.\n")
    else:
        append_only = False
        index_bargad = False
        print("✅ Keeping existing data, appending LFW only...\n")

confirm = input("⚠️ Continue with indexing? (y/n): ")
if confirm.lower() != "y":
    print("Cancelled.")
    exit()

# Model setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using: {DEVICE}\n")
mtcnn = MTCNN(image_size=160, margin=20)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

total = 0

with torch.no_grad():

    # ========== BARGAD EMPLOYEES ==========
    if index_bargad and os.path.isdir("dataset/bargad"):
        DATASET_DIR = "dataset/bargad"
        tracked_bargad = load_tracked(INDEXED_BARGAD_FILE) if append_only else set()
        if append_only:
            print(f"👥 Append-only Bargad (skip {len(tracked_bargad)} already indexed)...\n")
        else:
            print("👥 Indexing Bargad employees...\n")

        for person in tqdm(os.listdir(DATASET_DIR)):
            person_dir = os.path.join(DATASET_DIR, person)
            if not os.path.isdir(person_dir):
                continue

            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                key = f"{person}/{img_file}"
                if append_only and key in tracked_bargad:
                    continue

                img_path = os.path.join(person_dir, img_file)

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    h, w = img.shape[:2]
                    if max(h, w) > 800:
                        scale = 800 / max(h, w)
                        img = cv2.resize(img, (int(w * scale), int(h * scale)))

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face = mtcnn(img_rgb)
                    if face is None:
                        print(f"  ❌ No face: {img_file}")
                        continue

                    face = face.unsqueeze(0).to(DEVICE)
                    emb = model(face).cpu().numpy()[0].astype("float32")
                    emb = emb / np.linalg.norm(emb)

                    upload = cloudinary.uploader.upload(
                        img_path,
                        folder=f"facematch/bargad/{person}",
                        public_id=os.path.splitext(img_file)[0],
                        overwrite=True
                    )
                    image_url = upload["secure_url"]

                    insert_face_sync(
                        label=person,
                        source="bargad",
                        image_url=image_url,
                        embedding=emb.tolist(),
                    )

                    total += 1
                    append_tracked(INDEXED_BARGAD_FILE, key)
                    tracked_bargad.add(key)
                    print(f"  ✅ [{total}] {person}/{img_file} → Cloudinary ✓")

                except Exception as e:
                    print(f"  ❌ Error on {img_file}: {e}")
                    continue

        print(f"\n✅ Bargad done\n")
    elif index_bargad and not os.path.isdir("dataset/bargad"):
        print("⚠️ dataset/bargad not found, skipping Bargad.\n")

    # ========== LFW DATASET ==========
    LFW_DIR = "dataset/lfw-deepfunneled"
    MAX_LFW_IMAGES = 5000
    lfw_count = 0

    if os.path.isdir(LFW_DIR):
        tracked_lfw = load_tracked(INDEXED_LFW_FILE) if append_only else set()
        if append_only:
            print(f"🌍 Append-only LFW (skip {len(tracked_lfw)} already indexed)...\n")
        else:
            print("🌍 Indexing LFW dataset (max 5000 images, 2+ photos per person)...\n")

        for person in tqdm(os.listdir(LFW_DIR)):
            if not append_only and lfw_count >= MAX_LFW_IMAGES:
                break

            person_dir = os.path.join(LFW_DIR, person)
            if not os.path.isdir(person_dir):
                continue

            imgs = [f for f in os.listdir(person_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(imgs) < 2:
                continue  # skip single-photo people

            for img_file in imgs:
                if not append_only and lfw_count >= MAX_LFW_IMAGES:
                    break

                key = f"{person}/{img_file}"
                if append_only and key in tracked_lfw:
                    continue

                img_path = os.path.join(person_dir, img_file)

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    h, w = img.shape[:2]
                    if max(h, w) > 800:
                        scale = 800 / max(h, w)
                        img = cv2.resize(img, (int(w * scale), int(h * scale)))

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face = mtcnn(img_rgb)
                    if face is None:
                        print(f"  ❌ No face: {img_file}")
                        continue

                    face = face.unsqueeze(0).to(DEVICE)
                    emb = model(face).cpu().numpy()[0].astype("float32")
                    emb = emb / np.linalg.norm(emb)

                    upload = cloudinary.uploader.upload(
                        img_path,
                        folder=f"facematch/lfw/{person}",
                        public_id=os.path.splitext(img_file)[0],
                        overwrite=True
                    )
                    image_url = upload["secure_url"]

                    insert_face_sync(
                        label=person,
                        source="lfw",
                        image_url=image_url,
                        embedding=emb.tolist(),
                    )

                    lfw_count += 1
                    total += 1
                    append_tracked(INDEXED_LFW_FILE, key)
                    tracked_lfw.add(key)
                    if append_only:
                        print(f"  ✅ [{total}] {person}/{img_file} → Cloudinary ✓")
                    else:
                        print(f"  ✅ [{lfw_count}/5000] {person}/{img_file} → Cloudinary ✓")

                except Exception as e:
                    print(f"  ❌ Error on {img_file}: {e}")
                    continue

        print(f"\n✅ LFW done — {lfw_count} images indexed")
    else:
        if append_only:
            print("⚠️ dataset/lfw-deepfunneled not found, skipping LFW.\n")

print(f"\n🎉 Total — {total} faces indexed into PostgreSQL + Cloudinary")
