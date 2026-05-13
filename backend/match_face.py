# match_face.py
import cv2
import torch
import numpy as np
import faiss
import argparse
from facenet_pytorch import MTCNN, InceptionResnetV1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(image_size=160, margin=20)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

index = faiss.read_index("face_index.faiss")
labels = np.load("labels.npy", allow_pickle=True)

def match_face(image_path, top_k=5):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face = mtcnn(img)
    if face is None:
        print(" No face detected")
        return

    face = face.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(face).cpu().numpy().astype("float32")

    faiss.normalize_L2(emb)
    scores, idxs = index.search(emb, top_k)

    print("\n Top Matches:")
    for i in range(top_k):
        print(f"{labels[idxs[0][i]]}  |  Score: {scores[0][i]:.3f}")

# Example usage
if __name__ == "__main__":
    # test_image = "test.jpg"
    # match_face(test_image)
    parser = argparse.ArgumentParser(description="Face matching using FAISS")
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=5,
        help="Number of top matches to return (default: 5)"
    )

    args = parser.parse_args()
    match_face(args.image, args.top_k)