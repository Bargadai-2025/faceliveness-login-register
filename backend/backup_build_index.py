import os, cv2, numpy as np, torch, faiss
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1

DATASET_DIR = "dataset/bargad"  # ← only employee photos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

embeddings = []
labels = []

print("Processing employee photos...")

with torch.no_grad():
    for person in tqdm(os.listdir(DATASET_DIR)):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = mtcnn(img)
            if face is None:
                continue
            face = face.unsqueeze(0).to(DEVICE)
            emb = model(face).cpu().numpy()[0]
            embeddings.append(emb)
            labels.append(person)

if len(embeddings) == 0:
    print("ERROR: No faces detected. Check bargad folder structure.")
else:
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(512)
    index.add(embeddings)
    faiss.write_index(index, "face_index.faiss")
    np.save("labels.npy", labels)
    print(f"Done! Indexed {len(embeddings)} faces.")
