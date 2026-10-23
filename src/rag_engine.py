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

# Cache for retrieved passages and LLM outputs
cache = {}
embedding_cache = {}

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
    tokenized_corpus = [re.findall(r'[\u0E00-\u0E7F0-9a-zA-Z]+', text.lower()) for text in passage_texts]
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

def tokenize_text(text: str) -> list[str]:
    return re.findall(r'[\u0E00-\u0E7F0-9a-zA-Z]+', str(text).lower())

def calc_similarity(queries: list, texts: list, retriever_model, device) -> np.ndarray:
    cache_key = tuple(queries + texts)
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    if not queries or not texts:
        return np.zeros((len(queries), len(texts)))
    with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
        embeddings = retriever_model.encode(queries + texts, convert_to_numpy=True, device=device, batch_size=16)
        query_embs = embeddings[:len(queries)]
        text_embs = embeddings[len(queries):]
        scores = np.dot(query_embs, text_embs.T) / (np.linalg.norm(query_embs, axis=1)[:, None] * np.linalg.norm(text_embs, axis=1)[None, :] + 1e-6)
        for i, text in enumerate(texts):
            if any(kw in tokenize_text(text) for kw in LEGAL_KEYWORDS):
                scores[:, i] += 0.3
    embedding_cache[cache_key] = scores.clip(0, 1)
    return embedding_cache[cache_key]

def complete_last_sentence(text: str) -> str:
    if not text:
        return "."
    sentences = re.split(r'[.!?。]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return text + "."
    last_sentence = sentences[-1]
    if not last_sentence.endswith(('.','!','?','。')):
        return text + " ตามที่กฎหมายกำหนด."
    return text

# 3. RAG Core Functions
def retrieve_documents(query: str, collection, retriever_model, bm25_model, passages, device, k: int = 1):
    try:
        logger.info(f"Retrieving for query: {query}")
        tokenized_query = tokenize_text(query)
        bm25_scores = bm25_model.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[-100:]
        filtered_passages = [passages[i] for i in top_bm25_indices]
        with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
            query_emb = retriever_model.encode([query], convert_to_numpy=True, device=device).astype('float32')
        chroma_results = collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        results = []
        for doc, meta, dist in zip(chroma_results["documents"][0], chroma_results["metadatas"][0], chroma_results["distances"][0]):
            try:
                passage_idx = next(j for j, p in enumerate(passages) if p.get("text") == doc)
                dense_score = 1 - (dist / (max(chroma_results["distances"][0]) + 1e-6))
                sparse_score = bm25_scores[passage_idx] / (max(bm25_scores) + 1e-6 if max(bm25_scores) > 0 else 1e-6)
                combined_score = 0.8 * dense_score + 0.2 * sparse_score
                passage = passages[passage_idx]
                if any(kw in get_text(passage) for kw in LEGAL_KEYWORDS):
                    combined_score += 0.3
                results.append({"passage": passage, "score": combined_score})
            except StopIteration:
                continue
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
        logger.info(f"Found {len(sorted_results)} results, top score: {sorted_results[0]['score'] if sorted_results else 0}")
        return sorted_results
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []

def check_passage_relevance(question: str, passage_text: str, model, tokenizer, device) -> tuple[bool, bool]:
    try:
        prompt = (
            f"คำถาม: {question}\n"
            f"ข้อมูล: {passage_text}\n"
            f"ตอบเฉพาะ {{'is_relevant': bool, 'has_steps': bool}} โดย:\n"
            f"- 'is_relevant': true ถ้าข้อมูลเกี่ยวข้องกับคำถาม, false ถ้าไม่เกี่ยวข้อง\n"
            f"- 'has_steps': true ถ้าคำถามเกี่ยวกับวิธีการและข้อมูลมีขั้นตอนชัดเจน, false ถ้าไม่มี\n"
            f"ตัวอย่าง:\n"
            f"1. คำถาม: 'มัดจำคืออะไร' ข้อมูล: 'มัดจำนั้น ถ้ามิได้ตกลง...' → {{'is_relevant': true, 'has_steps': false}}\n"
            f"2. คำถาม: 'วิธีจัดตั้งบริษัท' ข้อมูล: 'การจัดตั้งต้องยื่น...' → {{'is_relevant': true, 'has_steps': true}}\n"
            f"3. คำถาม: 'มัดจำคืออะไร' ข้อมูล: 'สัญญาคือ...' → {{'is_relevant': false, 'has_steps': false}}\n"
            f"ตอบเฉพาะ JSON ในรูปแบบ {{}} เท่านั้น"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

        # ใช้ regex เพื่อดึงเฉพาะส่วนที่เป็น JSON
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                is_relevant = result.get('is_relevant', False)
                has_steps = result.get('has_steps', False)
                return is_relevant, has_steps
            except json.JSONDecodeError:
                logger.error(f"JSON decode error: {json_match.group(0)}")
                pass

    except Exception as e:
        logger.error(f"Relevance check error: {e}")
    
    # Fallback: ใช้ heuristic-based relevance checking
    question_lower = question.lower()
    passage_lower = passage_text.lower()
    
    # ตรวจสอบ keyword matching
    question_keywords = tokenize_text(question)
    passage_keywords = tokenize_text(passage_text)
    
    # นับคำที่ตรงกัน
    matching_keywords = set(question_keywords) & set(passage_keywords)
    
    is_relevant = len(matching_keywords) >= 2 or any(
        kw in passage_lower for kw in question_lower.split() if len(kw) > 2
    )
    
    has_steps = 'วิธี' in question_lower and any(
        step_word in passage_lower for step_word in ['ขั้นตอน', 'กระบวนการ', 'ต้อง', 'ดำเนินการ', 'จัดทำ']
    )
    
    logger.warning(f"Fallback to keyword-based relevance: is_relevant={is_relevant}, has_steps={has_steps}")
    return is_relevant, has_steps

def generate_llm_answer(question: str, context: str, passage: dict, score: float, model, tokenizer, device) -> str:
    if not model or not tokenizer:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"
    try:
        prompt = f"จากกฎหมายต่อไปนี้:\n{context}\n\nคำถาม: {question}\nคำตอบ:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
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
    
    # Generate LLM answer
    top_passage = retrieved[0]['passage']
    score = retrieved[0]['score']
    context = f"{get_meta(top_passage, 'law_title')} มาตรา {get_meta(top_passage, 'section')}: {get_text(top_passage)}"
    response = generate_llm_answer(question, context, top_passage, score, llm_model, llm_tokenizer, device)
    
    # ✅ แก้ไขส่วนนี้: แสดงข้อมูลแม้จะไม่ตรงเป๊ะ แต่มีข้อมูลใกล้เคียง
    if response == "ไม่พบข้อมูลที่เกี่ยวข้อง" and retrieved:
        # แสดงข้อมูลที่ใกล้เคียงที่สุดที่เจอ
        best_match = retrieved[0]
        best_text = get_text(best_match['passage'])
        best_law = get_meta(best_match['passage'], 'law_title', 'ไม่ทราบชื่อกฎหมาย')
        best_section = get_meta(best_match['passage'], 'section', 'ไม่ทราบมาตรา')
        
        output = f"💡 คำอธิบาย: ขออภัย ไม่พบคำตอบที่ตรงกับคำถามของคุณโดยเฉพาะ\n\n"
        output += f"📋 ข้อมูลกฎหมายที่เกี่ยวข้องที่พบ:\n"
        output += f"   📖 {best_law} มาตรา {best_section}\n"
        output += f"   📝 {best_text[:200]}..." if len(best_text) > 200 else f"   📝 {best_text}\n\n"
        output += f"   ⭐ คะแนนความเกี่ยวข้อง: {best_match['score']:.2f}\n\n"
        output += f"💡 คำแนะนำ: กรุณาระบุคำถามให้ชัดเจนยิ่งขึ้น หรือสอบถามข้อมูลเฉพาะเรื่อง"
    else:
        output = f"💡 คำอธิบาย: {response}\n"
        if response != "ไม่พบข้อมูลที่เกี่ยวข้อง":
            output += f"📚 แหล่งที่มา:\n   1. {get_meta(top_passage, 'law_title')} มาตรา {get_meta(top_passage, 'section')} (Score: {score:.2f})\n"
    
    output += f"⏱️ Time taken: {time.time() - start_time:.2f} seconds"
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