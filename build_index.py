import pickle
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ล้าง CUDA cache
torch.cuda.empty_cache()

# โหลด passages
with open("./data/passages.pkl", "rb") as f:
    passages = pickle.load(f)

print(f"Loaded {len(passages)} passages for indexing")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# โหลด retriever model
model = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device=device
)

# Encode passages
batch_size = 16
embeddings = []
ids = []
metadatas = []

print("Encoding passages:")
for i in tqdm(range(0, len(passages), batch_size)):
    batch = passages[i:i + batch_size]
    batch_texts = [p["text"] for p in batch]
    
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        batch_embeddings = model.encode(
            batch_texts,
            convert_to_numpy=True,
            device=device,
            batch_size=batch_size
        ).astype('float32')
    
    embeddings.extend(batch_embeddings)
    ids.extend([p["id"] for p in batch])
    metadatas.extend([p["metadata"] for p in batch])

# สร้าง Chroma collection
client = chromadb.PersistentClient(path="./data/chroma_db")
try:
    client.delete_collection(name="thai_legal")
except:
    pass
collection = client.create_collection(name="thai_legal")

# เพิ่ม passages เป็น batches
max_batch_size = 5000  # ปลอดภัยต่ำกว่า 5461
for i in tqdm(range(0, len(passages), max_batch_size), desc="Adding passages to Chroma"):
    batch_ids = ids[i:i + max_batch_size]
    batch_embeddings = embeddings[i:i + max_batch_size]
    batch_metadatas = metadatas[i:i + max_batch_size]
    batch_documents = [passages[j]["text"] for j in range(i, min(i + max_batch_size, len(passages)))]
    
    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas,
        documents=batch_documents
    )

print(f"Chroma collection created with {len(passages)} passages and saved to ./data/chroma_db")

# ล้าง memory
del model, embeddings, passages
torch.cuda.empty_cache()