# 📚 Legal RAG Chatbot for Thai Laws

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

**Legal RAG Chatbot** คือแชทบอทตอบคำถามกฎหมายไทย โดยใช้เทคนิค **Retrieval-Augmented Generation (RAG)** ซึ่งผสมผสานความสามารถของ *การค้นหา* และ *การสรุปด้วย AI*

## ✨ Features

- ✅ **แหล่งข้อมูลหลัก**: ประมวลกฎหมายแพ่งและพาณิชย์ + พระราชบัญญัติ (Acts)
- 🔎 **ระบบค้นหา**: Hybrid Search (Dense Retrieval จาก E5 + Sparse Retrieval จาก BM25)
- 🧠 **โมเดล**: ใช้ LLM ที่ Fine-tuned เพื่อการตอบที่แม่นยำ   ทอลองได้จาก DuckerMaster/thai-legal-lora  
- 🖥️ **UI Options**: รองรับทั้ง Command-Line และ Gradio GUI

## 🚀 Quick Start

### 📋 Prerequisites

- Python **3.9+**
- NVIDIA GPU + CUDA **11.8+** (แนะนำเพื่อประสิทธิภาพสูงสุด)

### ⚡ Installation

1. **Clone repository และสร้าง Virtual Environment**
   ```bash
   git clone <repository-url>
   cd legal-rag-chatbot
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

2. **ติดตั้ง Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **สร้างโฟลเดอร์ข้อมูล**
   ```bash
   mkdir data
   ```

> ⚠️ **หมายเหตุ**: หากพบข้อผิดพลาดเกี่ยวกับ CUDA ให้ตรวจสอบว่าติดตั้งไดรเวอร์ NVIDIA และ CUDA Toolkit เวอร์ชัน 11.8 ขึ้นไปแล้ว

## 📝 Step-by-Step Usage

### 1️⃣ การเตรียม Passage จาก Dataset

```bash
python data_preparation.py
```

- ⏱️ **เวลาที่ใช้**: 5-10 นาทีสำหรับข้อมูล 10,000 passages
- 📄 **ผลลัพธ์**: ไฟล์ `passages.pkl` จะถูกสร้างขึ้นในโฟลเดอร์ `./data`

### 2️⃣ การสร้าง Vector Database

```bash
python build_index.py
```

- ⏱️ **เวลาที่ใช้**: 10-20 นาที (ขึ้นอยู่กับประสิทธิภาพ GPU)
- 🗄️ **ผลลัพธ์**: ฐานข้อมูล Chroma จะถูกสร้างขึ้นในโฟลเดอร์ `./data/chroma_db`

> ⚠️ **หากพบปัญหา Out-of-Memory (OOM)**: ให้ลดค่า `batch_size` ในสคริปต์ `build_index.py`

### 3️⃣ การ Fine-tune LLM

```bash
python finetune_generator.py
```

- ⏱️ **เวลาที่ใช้**: 1-2 ชั่วโมงต่อ Epoch (แนะนำให้รัน 3 Epochs)
- 🎯 **ผลลัพธ์**: โมเดล LoRA ที่ผ่านการ Fine-tune จะถูกบันทึกในโฟลเดอร์ `./data/trained_lora_model/`

> ⚠️ **หากพบปัญหา OOM**: ให้ลดค่า `batch_size` เป็น 2 หรือเพิ่มค่า `gradient_accumulation_steps` เป็น 2

### 4️⃣ การใช้งาน Chatbot

#### 🖥️ Command-Line Interface

```bash
python rag_inference.py
```

- 💬 **วิธีใช้งาน**: พิมพ์คำถามเกี่ยวกับกฎหมายไทยเพื่อรับคำตอบ พิมพ์ `exit` เพื่อออกจากโปรแกรม
- 💾 **ข้อกำหนด**: ใช้การค้นหาแบบ Hybrid Search และเทคนิค 4-bit Quantization ที่ใช้ VRAM ประมาณ 4-6 GB

#### 🌐 Gradio GUI

```bash
python rag_gui.py
```

- 🌍 **วิธีใช้งาน**: เข้าถึง URL ที่ Gradio แสดงผล (โดยปกติคือ `http://127.0.0.1:7860`) เพื่อเริ่มการสนทนา
- 📦 **ข้อกำหนด**: ต้องติดตั้ง `gradio` เพิ่มเติม

## 🔧 Troubleshooting

| ปัญหา | วิธีแก้ไข |
|-------|----------|
| GPU ไม่ถูกตรวจพบ | รัน `python -c "import torch; print(torch.cuda.is_available())"` หากได้ผลลัพธ์เป็น `False` ให้ตรวจสอบการติดตั้งไดรเวอร์และการตั้งค่า CUDA |
| หน่วยความจำไม่เพียงพอ | ลดค่า `batch_size` ในสคริปต์ หรือเลือกใช้ข้อมูลย่อยๆ เช่น `ds["train"].select(range(2000))` |
| ข้อผิดพลาดเกี่ยวกับ ChromaDB | ลองติดตั้งใหม่ด้วยคำสั่ง `pip install chromadb --no-binary :all:` หรือติดตั้ง numpy เวอร์ชันที่เข้ากันได้ เช่น `pip install numpy==1.26.4` |

## 🏗️ Project Structure

```
legal-rag-chatbot/
├── data/                          # ข้อมูลและโมเดล
│   ├── passages.pkl              # ข้อมูล passages ที่ประมวลแล้ว
│   ├── chroma_db/               # Vector database
│   └── trained_lora_model/      # โมเดล LoRA ที่ fine-tune แล้ว
├── data_preparation.py          # สคริปต์เตรียมข้อมูล
├── build_index.py              # สคริปต์สร้าง vector database
├── finetune_generator.py       # สคริปต์ fine-tune โมเดล
├── rag_inference.py           # แชทบอทแบบ CLI
├── rag_gui.py                 # แชทบอทแบบ GUI
└── requirements.txt           # Dependencies
```


## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- dataset ข้อมูลกฎหมายไทย [airesearch/WangchanX-Legal-ThaiCCL-RAG]
- E5 embedding model
- ChromaDB และ BM25 สำหรับการค้นหา
- Hugging Face Transformers

---

💡 **Tips**: หากต้องการประสิทธิภาพสูงสุด แนะนำให้ใช้ GPU ที่มี VRAM อย่างน้อย 8GB
