import os
import sys
import cv2
import torch
import numpy as np
import argparse
from facenet_pytorch import MTCNN, InceptionResnetV1
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import asyncio

from database import close_db_pool, init_db_pool, insert_face

# Load environment variables
load_dotenv()

# Cloudinary Setup
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Model Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

async def add_user(image_path, name):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at {image_path}")
        return

    print(f"Processing image for {name}...")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    # Detect face and generate embedding
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = mtcnn(img_rgb)
    
    if face is None:
        print("Error: No face detected in the image.")
        return

    face = face.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(face).cpu().numpy()[0].astype("float32")
    emb = emb / np.linalg.norm(emb)

    # Upload to Cloudinary
    print("Uploading to Cloudinary...")
    upload = cloudinary.uploader.upload(
        image_path,
        folder=f"facematch/users/{name.replace(' ', '_')}",
        overwrite=True
    )
    image_url = upload["secure_url"]

    # Save to PostgreSQL
    print("Saving to PostgreSQL...")
    await insert_face(
        label=name,
        source="manual_reg",
        image_url=image_url,
        embedding=emb.tolist(),
    )
    print(f"Successfully registered {name}!")


async def main():
    parser = argparse.ArgumentParser(description="Add a new user to the Face Match database.")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--name", required=True, help="Name of the person")
    args = parser.parse_args()

    await init_db_pool()
    try:
        await add_user(args.image, args.name)
    finally:
        await close_db_pool()

if __name__ == "__main__":
    asyncio.run(main())
