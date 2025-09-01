# 📚 Legal RAG Chatbot for Thai Laws

**Legal RAG Chatbot** คือแชทบอทตอบคำถามกฎหมายไทย โดยใช้เทคนิค **Retrieval-Augmented Generation (RAG)** ซึ่งผสมผสานความสามารถของ *การค้นหา* และ *การสรุปด้วย AI*  

- ✅ แหล่งข้อมูลหลัก: **ประมวลกฎหมายแพ่งและพาณิชย์** + **พระราชบัญญัติ (Acts)**  
- 🔎 ระบบค้นหา: **Hybrid Search** (Dense Retrieval จาก E5 + Sparse Retrieval จาก BM25)  
- 🧠 โมเดล: ใช้ **LLM ที่ Fine-tuned** เพื่อการตอบที่แม่นยำ  

---

## 🚀 การติดตั้งและตั้งค่า (Setup)

### 📌 ข้อกำหนดเบื้องต้น (Prerequisites)
- Python **3.9+**
- NVIDIA GPU + CUDA **11.8+** (แนะนำเพื่อประสิทธิภาพสูงสุด)

---

### ⚙️ ขั้นตอนการติดตั้ง
1. **สร้าง Virtual Environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate


Bash

pip install -r requirements.txt
⚠️ หมายเหตุ: หากพบข้อผิดพลาดเกี่ยวกับ CUDA (CUDA error) ให้ตรวจสอบว่าติดตั้งไดรเวอร์ NVIDIA และ CUDA Toolkit เวอร์ชัน 11.8 ขึ้นไปแล้ว

สร้างโฟลเดอร์สำหรับจัดเก็บข้อมูล:

Bash

mkdir data
ขั้นตอนการรันโปรเจกต์ (Step-by-Step Guide)
ทำตามขั้นตอนด้านล่างนี้ตามลำดับเพื่อเตรียมความพร้อมของข้อมูลและโมเดล

1. การเตรียม Passage จาก Dataset
รันสคริปต์ data_preparation.py เพื่อดาวน์โหลดและประมวลผลข้อมูลดิบให้เป็น passages ที่พร้อมใช้งาน

Bash

python data_preparation.py
เวลาที่ใช้โดยประมาณ: 5-10 นาทีสำหรับข้อมูล 10,000 passages

ผลลัพธ์: ไฟล์ passages.pkl จะถูกสร้างขึ้นในโฟลเดอร์ ./data

2. การสร้าง Vector Database
รันสคริปต์ build_index.py เพื่อสร้าง Vector Database ด้วย ChromaDB และ BM25 สำหรับการค้นหา

Bash

python build_index.py
เวลาที่ใช้โดยประมาณ: 10-20 นาที (ขึ้นอยู่กับประสิทธิภาพ GPU ในการเข้ารหัสข้อมูล)

ผลลัพธ์: ฐานข้อมูล Chroma จะถูกสร้างขึ้นในโฟลเดอร์ ./data/chroma_db

⚠️ หากพบปัญหา Out-of-Memory (OOM): ให้ลดค่า batch_size ในสคริปต์ build_index.py

3. การ Fine-tune LLM
รันสคริปต์ finetune_generator.py เพื่อปรับแต่ง LLM ให้สามารถสร้างคำตอบตามรูปแบบที่ต้องการ

Bash

python finetune_generator.py
เวลาที่ใช้โดยประมาณ: 1-2 ชั่วโมงต่อ Epoch (แนะนำให้รัน 3 Epochs)

ผลลัพธ์: โมเดล LoRA ที่ผ่านการ Fine-tune จะถูกบันทึกในโฟลเดอร์ ./data/trained_lora_model/

⚠️ หากพบปัญหา OOM: ให้ลดค่า batch_size เป็น 2 หรือเพิ่มค่า gradient_accumulation_steps เป็น 2

4. การใช้งาน Chatbot
คุณสามารถเลือกใช้งานแชทบอทได้ 2 รูปแบบ คือแบบ Command-Line และแบบ GUI

4.1 การใช้งานแบบ Command-Line
รันสคริปต์ rag_inference.py เพื่อเรียกใช้งานแชทบอทผ่าน Command Line

Bash

python rag_inference.py
วิธีใช้งาน: พิมพ์คำถามเกี่ยวกับกฎหมายไทยเพื่อรับคำตอบ พิมพ์ exit เพื่อออกจากโปรแกรม

ข้อกำหนด: ใช้การค้นหาแบบ Hybrid Search และเทคนิค 4-bit Quantization ที่ใช้ VRAM ประมาณ 4-6 GB

4.2 การใช้งานแบบ Graphical User Interface (GUI)
รันสคริปต์ rag_gui.py เพื่อเรียกใช้งานแชทบอทผ่าน Gradio UI ที่สวยงามและใช้งานง่าย

Bash

python rag_gui.py
วิธีใช้งาน: เข้าถึง URL ที่ Gradio แสดงผล (โดยปกติคือ http://127.0.0.1:7860) เพื่อเริ่มการสนทนา

ข้อกำหนด: ต้องติดตั้ง gradio เพิ่มเติม

การแก้ไขปัญหา (Troubleshooting)
ปัญหา	วิธีแก้ไข
GPU ไม่ถูกตรวจพบ	รัน python -c "import torch; print(torch.cuda.is_available())" หากได้ผลลัพธ์เป็น False ให้ตรวจสอบการติดตั้งไดรเวอร์และการตั้งค่า CUDA
หน่วยความจำไม่เพียงพอ	ลดค่า batch_size ในสคริปต์ หรือเลือกใช้ข้อมูลย่อยๆ เช่น ds["train"].select(range(2000))
ข้อผิดพลาดเกี่ยวกับ ChromaDB	ลองติดตั้งใหม่ด้วยคำสั่ง pip install chromadb --no-binary :all: หรือติดตั้ง numpy เวอร์ชันที่เข้ากันได้ เช่น pip install numpy==1.26.4



