import os
import pickle
from collections import defaultdict
from datasets import load_dataset
import torch

# ล้าง CUDA cache เพื่อประหยัด VRAM
torch.cuda.empty_cache()

# สร้างโฟลเดอร์ data ถ้ายังไม่มี
os.makedirs("./data", exist_ok=True)

# โหลด dataset ด้วย streaming เพื่อลด RAM usage
ds = load_dataset("airesearch/WangchanX-Legal-ThaiCCL-RAG", streaming=True)

# สร้าง unique passages สำหรับ Chroma
unique_passages = defaultdict(dict)
for split in ['train', 'test']:
    for rec in ds[split]:
        for pc in rec['positive_contexts']:
            key = pc['unique_key']
            if key not in unique_passages:
                # สร้าง format ที่เหมาะกับ Chroma
                unique_passages[key] = {
                    "id": f"passage_{key}",  # ID เฉพาะ (string) สำหรับ Chroma
                    "text": pc['context'].strip(),  # ลบช่องว่างส่วนเกิน
                    "metadata": {
                        "original_key": key,  # เก็บ key เดิม
                        "category": pc['metadata'].get('category', 'ThaiCCL'),  # เพิ่ม category ถ้ามี
                        "source": "WangchanX-Legal-ThaiCCL-RAG",
                        # เพิ่ม field อื่นจาก metadata ถ้ามี เช่น section, law_type
                        **{k: v for k, v in pc['metadata'].items() if k != 'category'}
                    }
                }

# แปลงเป็น list
passages = list(unique_passages.values())
print(f"Unique passages created: {len(passages)}")

# บันทึก passages
with open("./data/passages.pkl", "wb") as f:
    pickle.dump(passages, f)
print("Passages saved to ./data/passages.pkl")

# ล้าง memory
del unique_passages, ds
torch.cuda.empty_cache()