import os, cv2, numpy as np, torch, faiss
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1

DATASET_DIR = "dataset/bargad"
INDEX_FILE = "face_index.faiss"
LABELS_FILE = "labels.npy"
PROCESSED_FILE = "processed_images.txt"  # tracks already processed photos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# ── Load existing index and labels if they exist ──
if os.path.exists(INDEX_FILE) and os.path.exists(LABELS_FILE):
    print("Loading existing index...")
    index = faiss.read_index(INDEX_FILE)
    labels = list(np.load(LABELS_FILE, allow_pickle=True))
    print(f"Existing index has {index.ntotal} faces.")
else:
    print("No existing index found. Building from scratch...")
    index = faiss.IndexFlatIP(512)
    labels = []

# ── Load list of already processed images ──
if os.path.exists(PROCESSED_FILE):
    with open(PROCESSED_FILE, "r") as f:
        processed = set(f.read().splitlines())
else:
    processed = set()

# ── Find only NEW images ──
new_images = []
for person in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_dir):
        continue
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        if img_path not in processed:
            new_images.append((img_path, person))

if len(new_images) == 0:
    print("No new photos found. Index is already up to date!")
    exit()

print(f"Found {len(new_images)} new photo(s) to process...")

# ── Process only new images ──
new_embeddings = []
new_labels = []
newly_processed = []

with torch.no_grad():
    for img_path, person in tqdm(new_images):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(img)
        if face is None:
            print(f"  No face detected: {img_path}")
            continue
        face = face.unsqueeze(0).to(DEVICE)
        emb = model(face).cpu().numpy()[0]
        new_embeddings.append(emb)
        new_labels.append(person)
        newly_processed.append(img_path)

if len(new_embeddings) == 0:
    print("No faces detected in new photos. Check image quality.")
else:
    # ── Add new embeddings to existing index ──
    new_embeddings = np.array(new_embeddings).astype("float32")
    faiss.normalize_L2(new_embeddings)
    index.add(new_embeddings)
    labels.extend(new_labels)

    # ── Save updated index and labels ──
    faiss.write_index(index, INDEX_FILE)
    np.save(LABELS_FILE, labels)

    # ── Save processed image paths ──
    with open(PROCESSED_FILE, "a") as f:
        f.write("\n".join(newly_processed) + "\n")

    print(f"Done! Added {len(new_embeddings)} new faces.")
    print(f"Total faces in index: {index.ntotal}")
