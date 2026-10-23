import pickle
import chromadb
import numpy as np
import torch
import re
import sys
import logging
import os
import time
import json
from collections import OrderedDict
from pythainlp.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from peft import PeftModel
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ปิด DEBUG_MODE เพื่อไม่ให้แสดง log ของการดีบั๊ก
DEBUG_MODE = False
if DEBUG_MODE:
    logging.getLogger().setLevel(logging.DEBUG)

# Global legal keywords
LEGAL_KEYWORDS = {
    'บริษัท', 'หุ้น', 'กรรมการ', 'กฎหมาย', 'มาตรา', 'พระราชบัญญัติ', 'จดทะเบียน', 'ประชุม', 'ภาษี', 'คืนภาษี', 'ยื่นคำร้อง',
    'แพ่ง', 'พาณิชย์', 'รัษฎากร', 'สินทรัพย์', 'ดิจิทัล', 'นิติบุคคล', 'กองทุน', 'สำรอง', 'เลี้ยงชีพ', 'หุ้นส่วน',
    'รัฐมนตรี', 'จัดซื้อ', 'จัดจ้าง', 'พัสดุ', 'ภาครัฐ', 'บัญชี', 'พลังงาน', 'คนต่างด้าว', 'รัฐวิสาหกิจ', 'อนุรักษ์',
    'ห้างหุ้นส่วน', 'สมาคม', 'มูลนิธิ', 'พนักงาน', 'ทรัสต์', 'ธุรกรรม', 'ตลาดทุน', 'ทะเบียน', 'ทุน', 'ธุรกิจ',
    'สถาบันการเงิน', 'ธนาคารแห่งประเทศไทย', 'ธุรกิจสถาบันการเงิน', 'ธนาคารพาณิชย์', 'เครดิตฟองซิเอร์', 'หลักทรัพย์',
    'หุ้นสามัญ', 'หุ้นบุริมสิทธิ', 'ผู้ถือหุ้น', 'การถือหุ้น', 'เงินปันผล', 'บริษัทจำกัด', 'บริษัทมหาชนจำกัด', 'บริษัทลูก',
    'ใบอนุญาต', 'เงินกองทุน', 'การให้สินเชื่อ', 'การรับฝากเงิน', 'การระดมเงิน', 'พระราชกฤษฎีกา', 'ราชกิจจานุเบกษา',
    'การกำกับดูแล', 'การบริหารงาน', 'ความเห็นชอบ', 'การตรวจสอบ', 'สำนักงานใหญ่', 'สาขา', 'อัตราดอกเบี้ย', 'ค่าบริการ',
    'ความเสี่ยง', 'นิติกรรม', 'สัญญา', 'ทุจริต', 'ป้องกัน', 'ปราบปราม', 'เงินได้', 'ปิโตรเลียม', 'ซื้อขาย', 'สินค้าเกษตร',
    'ล่วงหน้า', 'วิชาชีพ', 'งบประมาณ', 'วินัย', 'การเงิน', 'คลัง', 'องค์การ', 'ความผิด', 'ข้าราชการ',
    'พัฒนา', 'เศรษฐกิจ', 'สังคม', 'หลักประกัน', 'หอการค้า', 'แรงงาน', 'สัมพันธ์', 'ป.ป.ช.', 'แสตมป์', 'ใบกำกับภาษี',
    'ธุรกิจเฉพาะ', 'อนุรักษ์พลังงาน', 'คุณสมบัติมาตรฐาน', 'การบัญชี', 'สภาพัฒนาการ', 'แรงงานรัฐวิสาหกิจ', 'วินัยข้าราชการ',
    'พระราชกำหนด', 'แปลงสินทรัพย์', 'หลักทรัพย์และตลาดหลักทรัพย์', 'ประกอบกิจการพลังงาน', 'ประกอบธุรกิจคนต่างด้าว',
    'พัฒนาการกำกับดูแลรัฐวิสาหกิจ', 'ส่งเสริมอนุรักษ์エネルギー', 'กำหนดความผิดห้างหุ้นส่วน', 'คุณสมบัติกรรมการรัฐวิสาหกิจ',
    'ทรัสต์ตลาดทุน', 'ทะเบียนพาณิชย์', 'ทุนรัฐวิสาหกิจ', 'บริษัทมหาชนจำกัด', 'ป้องกันปราบปรามทุจริต',
    'ภาษีเงินได้ปิโตรเลียม', 'ยกเลิกซื้อขายสินค้าเกษตรล่วงหน้า', 'วิชาชีพบัญชี', 'วิธีการงบประมาณ', 'วินัยการเงินการคลัง',
    'จัดตั้งองค์การรัฐบาล', 'ความผิดพนักงานองค์การรัฐ', 'ความผิดวินัยข้าราชการ', 'สภาพัฒนาเศรษฐกิจสังคม', 'สมาคมการค้า',
    'สัญญาซื้อขายล่วงหน้า', 'หลักทรัพย์ตลาดหลักทรัพย์', 'หลักประกันธุรกิจ', 'หอการค้า', 'แรงงานรัฐวิสาหกิจสัมพันธ์',
    'บริหาร', 'จัดการ', 'หน่วยงาน', 'ข้อกำหนด', 'สิทธิ', 'หน้าที่', 'บทลงโทษ', 'การปฏิบัติ', 'องค์กร', 'การจัดตั้ง', 'มัดจำ'
}

# Bounded cache to prevent memory leaks in long-running services
class BoundedCache(OrderedDict):
    def __init__(self, maxsize=500, *args, **kwargs):
        self.maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if len(self) >= self.maxsize:
            self.popitem(last=False)
        super().__setitem__(key, value)

cache = BoundedCache(maxsize=500)
embedding_cache = BoundedCache(maxsize=500)

# 1. Configuration & Model Loading
def load_models_and_data(model_dir: str = "./data"):
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cpu":
        logger.warning("GPU not available, falling back to CPU with full-precision model. Performance may be slower.")

    try:
        client = chromadb.PersistentClient(path=f"{model_dir}/chroma_db")
        collection = client.get_collection(name="thai_legal")
    except Exception as e:
        logger.error(f"Error loading ChromaDB collection: {e}")
        return (None,) * 7

    try:
        with open(f"{model_dir}/passages.pkl", "rb") as f:
            passages = pickle.load(f)
        logger.info(f"Loaded {len(passages)} passages.")
    except Exception as e:
        logger.error(f"Error loading passages file: {e}")
        return (None,) * 7

    retriever_model = SentenceTransformer("intfloat/multilingual-e5-small", device=device)
    passage_texts = [p["text"] for p in passages]
    logger.info("Building BM25 index with PyThaiNLP word tokenization...")
    tokenized_corpus = [word_tokenize(text.lower(), engine="newmm") for text in passage_texts]
    bm25_model = BM25Okapi(tokenized_corpus)

    llm_model, llm_tokenizer = None, None
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/trained_lora_model", local_files_only=True)
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        if device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                "scb10x/llama3.2-typhoon2-3b",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                quantization_config=quantization_config
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                "scb10x/llama3.2-typhoon2-3b",
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
        llm_model = PeftModel.from_pretrained(base_model, f"{model_dir}/trained_lora_model")
        dummy_input = llm_tokenizer("สวัสดี", return_tensors="pt").to(device)
        _ = llm_model.generate(**dummy_input, max_new_tokens=10)
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        return (None,) * 7

    return passages, collection, retriever_model, bm25_model, llm_model, llm_tokenizer, device

# 2. Utility Functions
def get_text(passage: dict) -> str:
    return passage.get('text', '') if isinstance(passage, dict) else str(passage)

def get_meta(passage: dict, key: str, default: any = '') -> any:
    meta = passage.get('metadata', {}) if isinstance(passage, dict) else {}
    return meta.get(key, default) if isinstance(meta, dict) else default

# BM25 Stopwords to prevent lexical traps on procedural/question phrasing
BM25_STOPWORDS = {
    'เงื่อนไข', 'อย่างไร', 'ขั้นตอน', 'มีอะไรบ้าง', 'ข้อกำหนด', 'คืออะไร', 'กรณีใด',
    'อะไร', 'ทำไม', 'อย่างไรบ้าง', 'มีผลอย่างไร', 'ต้องทำอย่างไร', 'หรือไม', 'หรือไม่',
    'การ', 'ความ', 'มี', 'ได้', 'ต้อง', 'ให้', 'ตาม', 'ใน', 'ของ', 'และ', 'หรือ', 'ถ้า',
    'ผู้', 'ที่', 'ซึ่ง', 'อัน', 'ไว้', 'แก่', 'ต่อ', 'ด้วย', 'แห่ง', 'นั้น', 'นี้',
    'เป็นอย่างไร', 'หมายความว่าอย่างไร'
}

def tokenize_text(text: str) -> list[str]:
    return word_tokenize(str(text).lower(), engine="newmm")

def tokenize_query_bm25(text: str) -> list[str]:
    tokens = word_tokenize(str(text).lower(), engine="newmm")
    filtered = [t for t in tokens if t.strip() and t not in BM25_STOPWORDS]

    # Compound sub-segmentation for legal terms to prevent dictionary mismatch
    expanded = list(filtered)
    legal_components = [
        'ครอบครอง', 'ปรปักษ์', 'นิติกรรม', 'สัญญา', 'มัดจำ', 'หุ้นส่วน', 'บริษัท',
        'กรรมการ', 'ประชุม', 'จดทะเบียน', 'ละเมิด', 'ทรัพย์สิน', 'หนี้', 'กรรมสิทธิ์',
        'ผู้ถือหุ้น', 'เงินปันผล', 'ภาษี', 'หลักประกัน'
    ]
    for t in filtered:
        for comp in legal_components:
            if comp in t and comp != t and comp not in expanded:
                expanded.append(comp)

    clean = [x for x in expanded if x.strip() and x not in BM25_STOPWORDS]
    return clean if clean else [t for t in tokens if t.strip()]

# 3. RAG Core Functions
def retrieve_documents(query: str, collection, retriever_model, bm25_model, passages, device, k: int = 1):
    try:
        logger.info(f"Retrieving for query: {query}")
        tokenized_query = tokenize_query_bm25(query)
        bm25_scores = np.array(bm25_model.get_scores(tokenized_query), dtype=np.float32)
        max_bm25 = float(bm25_scores.max()) if len(bm25_scores) > 0 and bm25_scores.max() > 0 else 0.0

        n_candidates = min(100, len(passages))
        top_bm25_indices = set(np.argsort(bm25_scores)[-n_candidates:])

        # Query embedding with E5 query: prefix
        query_text = f"query: {query}" if not query.startswith("query: ") else query
        with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
            query_emb = retriever_model.encode([query_text], convert_to_numpy=True, device=device).astype('float32')

        chroma_results = collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"]
        )

        dense_dict = {}
        if chroma_results and chroma_results.get("documents") and chroma_results["documents"][0]:
            for doc, dist in zip(chroma_results["documents"][0], chroma_results["distances"][0]):
                dense_sim = max(0.0, 1.0 - (dist / 2.0))
                dense_dict[doc] = dense_sim

        # Fast lookup mapping document text to index
        doc_to_idx = {p.get("text", ""): idx for idx, p in enumerate(passages)}
        candidate_indices = set(top_bm25_indices)
        for doc in dense_dict:
            if doc in doc_to_idx:
                candidate_indices.add(doc_to_idx[doc])

        results = []
        for idx in candidate_indices:
            passage = passages[idx]
            doc_text = passage.get("text", "")

            # Normalized BM25 score
            sparse_score = (bm25_scores[idx] / max_bm25) if max_bm25 > 0 else 0.0
            dense_score = dense_dict.get(doc_text, 0.0)

            # Combined score: 50% Dense + 50% Sparse
            combined_score = 0.5 * dense_score + 0.5 * sparse_score

            # Subtle keyword boost for legal terms
            if any(kw in doc_text for kw in LEGAL_KEYWORDS):
                combined_score += 0.05

            results.append({"passage": passage, "score": float(combined_score)})

        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
        logger.info(f"Found {len(sorted_results)} results, top score: {sorted_results[0]['score'] if sorted_results else 0:.4f}")
        return sorted_results
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []

def generate_llm_answer(question: str, context: str, passage: dict, score: float, model, tokenizer, device) -> str:
    if not model or not tokenizer or not context or not context.strip():
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"
    try:
        prompt = f"จากกฎหมายต่อไปนี้:\n{context}\n\nคำถาม: {question}\nคำตอบ:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "คำตอบ:" in response:
            answer = response.split("คำตอบ:")[-1].strip()
        else:
            answer = response[len(prompt):].strip()
        return answer if answer else get_text(passage)
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"ข้อมูลที่เกี่ยวข้อง: {get_text(passage)[:300]}..."
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_answer(question: str, passages, collection, retriever_model, bm25_model, llm_model, llm_tokenizer, device):
    start_time = time.time()
    
    # แก้ไข validation - อนุญาตคำถามสั้นแต่มีความหมาย
    cleaned_question = question.strip()
    if len(cleaned_question) < 3:
        return f"💡 คำอธิบาย: กรุณาระบุคำถามให้ชัดเจนยิ่งขึ้น\n⏱️ Time taken: {time.time() - start_time:.2f} seconds", []
    
    # ตรวจสอบว่ามีคำสำคัญทางกฎหมายหรือไม่
    thai_chars = sum(1 for c in cleaned_question if '\u0E00' <= c <= '\u0E7F')
    if thai_chars < 1 and not any(kw in cleaned_question.lower() for kw in LEGAL_KEYWORDS):
        return f"💡 คำอธิบาย: กรุณาระบุคำถามเป็นภาษาไทยหรือใช้คำศัพท์ทางกฎหมาย\n⏱️ Time taken: {time.time() - start_time:.2f} seconds", []
    
    if question in cache:
        retrieved, cached_response = cache[question]
        if cached_response:
            return f"{cached_response}\n⏱️ Time taken: {time.time() - start_time:.2f} seconds", retrieved
    else:
        retrieved = retrieve_documents(question, collection, retriever_model, bm25_model, passages, device, k=3)  # เพิ่ม k=3
        cache[question] = (retrieved, None)
    
    if not retrieved:
        return f"💡 คำอธิบาย: ไม่พบข้อมูลที่เกี่ยวข้อง\n⏱️ Time taken: {time.time() - start_time:.2f} seconds", []
    
    # Check confidence threshold to prevent hallucination
    top_passage = retrieved[0]['passage']
    score = retrieved[0]['score']
    MIN_CONFIDENCE_THRESHOLD = 0.28

    if score < MIN_CONFIDENCE_THRESHOLD:
        output = (
            f"💡 คำอธิบาย: ขออภัย ไม่พบบทบัญญัติกฎหมายที่เกี่ยวข้องโดยตรงกับคำถามนี้ในคลังข้อมูล "
            f"(คะแนนความเกี่ยวข้องต่ำกว่าเกณฑ์: {score:.2f})\n\n"
            f"💡 คำแนะนำ: กรุณาระบุคำถามให้เฉพาะเจาะจง หรือระบุเลขมาตรา/ชื่อ พ.ร.บ. ธุรกิจที่ต้องการค้นหา\n"
            f"⏱️ Time taken: {time.time() - start_time:.2f} seconds"
        )
        cache[question] = (retrieved, output)
        return output, retrieved

    # Generate LLM answer
    passage_text = get_text(top_passage).strip()
    law_title = get_meta(top_passage, 'law_title', '')
    section = str(get_meta(top_passage, 'section', ''))
    if law_title and law_title not in passage_text[:80]:
        context = f"{law_title} มาตรา {section}: {passage_text}"
    else:
        context = passage_text
    response = generate_llm_answer(question, context, top_passage, score, llm_model, llm_tokenizer, device)
    
    if response == "ไม่พบข้อมูลที่เกี่ยวข้อง":
        output = (
            f"💡 คำอธิบาย: ขออภัย ระบบไม่สามารถสกัดสรุปคำตอบจากตัวบทกฎหมายที่พบได้อย่างสมบูรณ์\n\n"
            f"💡 คำแนะนำ: กรุณาระบุประเด็นคำถามให้ชัดเจนยิ่งขึ้น\n"
        )
    else:
        output = f"💡 คำอธิบาย: {response}\n\n"

    output += "📚 แหล่งที่มาที่เกี่ยวข้อง:\n"
    for rank, item in enumerate(retrieved[:3], 1):
        p = item['passage']
        output += f"   {rank}. {get_meta(p, 'law_title')} มาตรา {get_meta(p, 'section')} (ความเกี่ยวข้อง: {item['score']:.2f})\n"
    
    output += f"\n⏱️ Time taken: {time.time() - start_time:.2f} seconds"
    cache[question] = (retrieved, output)
    return output, retrieved

# 4. Main Application & User Interface
def main():
    passages, collection, retriever_model, bm25_model, llm_model, llm_tokenizer, device = load_models_and_data()
    if not all([passages, collection, retriever_model, bm25_model, llm_model, llm_tokenizer]):
        logger.error("Application cannot run due to failed component loading.")
        sys.exit(1)

    print("🧪 ทดสอบระบบ (คำตอบ)")
    test_questions = [
        # "หน้าที่ของกรรมการ",
        # "การยื่นภาษีเงินได้ปิโตรเลียมต้องทำอย่างไร",
        # "วิธีจัดตั้งหน่วยงานใหม่ของรัฐต้องทำอย่างไร",
        # "หน้าที่ปปช",
        # "สมาคมการค้าคืออะไร"
        "การประชุม",
        "วิธีป้องกันทุจริตในหน่วยงานของรัฐ",
    ]
    for i, q in enumerate(test_questions, 1):
        print(f"\n❓ คำถามที่ {i}: {q}")
        resp, sources = get_answer(q, passages, collection, retriever_model, bm25_model, llm_model, llm_tokenizer, device)
        print(resp)
        print("---")
    
    print("🤖 บอทตอบคำถามกฎหมายไทย")
    print("💡 ตัวอย่าง: บริษัท, หุ้น, กรรมการ, การประชุม, ภาษี, ทุจริต, รัฐวิสาหกิจ, พลังงาน, สมาคม, บัญชี, การบริหารจัดการ, มัดจำ, การเสียภาษีอากร")
    print("พิมพ์ 'exit' เพื่อออก\n" + "="*50)
    while True:
        user_input = input("👤 ถามกฎหมาย: ").strip()
        if user_input.lower() in ["exit", "ออก", "quit"]:
            print("ขอบคุณที่ใช้บริการ 😊")
            break
        if not user_input:
            continue
        resp, sources = get_answer(user_input, passages, collection, retriever_model, bm25_model, llm_model, llm_tokenizer, device)
        print("\n" + resp)
        print("-"*50)

if __name__ == "__main__":
    main()