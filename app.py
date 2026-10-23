import gradio as gr
import pickle
import chromadb
import numpy as np
import torch
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from peft import PeftModel
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize
import time
import os
import sys

# Legal keywords
LEGAL_KEYWORDS = {
    'บริษัท', 'หุ้น', 'กรรมการ', 'กฎหมาย', 'มาตรา', 'ประชุม', 'ภาษี', 'ทุจริต', 'ป.ป.ช.', 'รัฐวิสาหกิจ',
    'สัญญา', 'นิติกรรม', 'มัดจำ', 'จดทะเบียน', 'หุ้นส่วน', 'บัญชี', 'พลังงาน', 'คนต่างด้าว', 'อนุรักษ์',
    'สมาคม', 'มูลนิธิ', 'พนักงาน', 'ทรัสต์', 'ทะเบียน', 'ทุน', 'ธุรกิจ', 'สถาบันการเงิน', 'หลักทรัพย์',
    'ผู้ถือหุ้น', 'เงินปันผล', 'ใบอนุญาต', 'การบริหาร', 'ความเสี่ยง', 'ป้องกัน', 'ปราบปราม', 'เงินได้',
    'วิชาชีพ', 'งบประมาณ', 'วินัย', 'การเงิน', 'คลัง', 'องค์การ', 'ความผิด', 'ข้าราชการ', 'พัฒนา',
    'เศรษฐกิจ', 'หลักประกัน', 'หอการค้า', 'แรงงาน', 'สัมพันธ์'
}

# Legal categories (35 Acts in Thai Corporate & Commercial Law Corpus)
LEGAL_CATEGORIES = {
    "ประมวลกฎหมายแพ่งและพาณิชย์": "สัญญา, นิติกรรม, มัดจำ, ทรัพย์สิน, ครอบครองปรปักษ์, ห้างหุ้นส่วน, บริษัท",
    "ประมวลรัษฎากร": "ภาษีเงินได้บุคคลธรรมดา/นิติบุคคล, ภาษีมูลค่าเพิ่ม (VAT), ภาษีธุรกิจเฉพาะ, อากรแสตมป์",
    "พระราชบัญญัติบริษัทมหาชนจำกัด พ.ศ. 2535": "การจัดตั้ง, ประชุมผู้ถือหุ้น, บอร์ดกรรมการ, หุ้นและงบการเงิน",
    "พระราชบัญญัติหลักทรัพย์และตลาดหลักทรัพย์ พ.ศ. 2535": "การกำกับดูแลตลาดทุน, การเสนอขายหุ้น, ก.ล.ต.",
    "พระราชบัญญัติธุรกิจสถาบันการเงิน พ.ศ. 2551": "ธนาคารพาณิชย์, เงินกองทุน, สินเชื่อ, การประกอบธุรกิจการเงิน",
    "พระราชบัญญัติประกอบรัฐธรรมนูญว่าด้วยการป้องกันและปราบปรามการทุจริต พ.ศ. 2561": "อำนาจหน้าที่ ป.ป.ช., การป้องกันการทุจริตในภาครัฐ",
    "พระราชบัญญัติการประกอบธุรกิจของคนต่างด้าว พ.ศ. 2542": "การควบคุมคนต่างด้าวประกอบธุรกิจ, ใบอนุญาต, บัญชีท้าย พ.ร.บ.",
    "พระราชบัญญัติการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560": "วิธีจัดซื้อจัดจ้างภาครัฐ, ระเบียบพัสดุ, สัญญาภาครัฐ",
    "พระราชบัญญัติการบัญชี พ.ศ. 2543": "การจัดทำบัญชี, ผู้มีหน้าที่จัดทำบัญชี, ผู้ทำบัญชี, งบการเงิน",
    "พระราชบัญญัติวิชาชีพบัญชี พ.ศ. 2547": "สภาวิชาชีพบัญชี, มาตรฐานการสอบบัญชี, ผู้สอบบัญชีรับอนุญาต (CPA)",
    "พระราชบัญญัติหลักประกันทางธุรกิจ พ.ศ. 2558": "สัญญาหลักประกันทางธุรกิจ, ผู้บังคับหลักประกัน, การจดทะเบียน",
    "พระราชบัญญัติการพัฒนาการกำกับดูแลและบริหารรัฐวิสาหกิจ พ.ศ. 2562": "การบริหารรัฐวิสาหกิจ, แผนพัฒนารัฐวิสาหกิจ, คณะกรรมการ คนร.",
    "พระราชบัญญัติการส่งเสริมการอนุรักษ์พลังงาน พ.ศ. 2535": "โรงงานควบคุม, อาคารควบคุม, กองทุนเพื่อส่งเสริมการอนุรักษ์พลังงาน",
    "พระราชบัญญัติการประกอบกิจการพลังงาน พ.ศ. 2550": "การกำกับดูแลไฟฟ้าและก๊าซธรรมชาติ, คณะกรรมการกำกับกิจการพลังงาน",
    "พระราชกำหนดการประกอบธุรกิจสินทรัพย์ดิจิทัล พ.ศ. 2561": "คริปโทเคอร์เรนซี, โทเคนดิจิทัล, ศูนย์ซื้อขายสินทรัพย์ดิจิทัล",
    "พระราชกำหนดนิติบุคคลเฉพาะกิจเพื่อการแปลงสินทรัพย์เป็นหลักทรัพย์ พ.ศ. 2540": "การแปลงสินทรัพย์เป็นหลักทรัพย์ (Securitization), SPV",
    "พระราชบัญญัติกองทุนสำรองเลี้ยงชีพ พ.ศ. 2530": "การจัดตั้งกองทุนสำรองเลี้ยงชีพ, คณะกรรมการกองทุน, เงินสะสม/สมทบ",
    "พระราชบัญญัติทรัสต์เพื่อธุรกรรมในตลาดทุน พ.ศ. 2550": "ทรัสตี, ทรัสต์เพื่อการลงทุนในอสังหาริมทรัพย์ (REIT)",
    "พระราชบัญญัติทะเบียนพาณิชย์ พ.ศ. 2499": "การจดทะเบียนพาณิชย์ของบุคคลธรรมดาและนิติบุคคล",
    "พระราชบัญญัติสัญญาซื้อขายล่วงหน้า พ.ศ. 2546": "สัญญาซื้อขายล่วงหน้า, ตัวแทนและผู้จัดการซื้อขายล่วงหน้า",
    "พระราชบัญญัติหอการค้า พ.ศ. 2509": "การจัดตั้งและการดำเนินงานของหอการค้า",
    "พระราชบัญญัติสมาคมการค้า พ.ศ. 2509": "การจัดตั้งและการดำเนินงานของสมาคมการค้า",
    "พระราชบัญญัติแรงงานรัฐวิสาหกิจสัมพันธ์ พ.ศ. 2543": "สิทธิลูกจ้างรัฐวิสาหกิจ, สภาพการจ้าง, สหภาพแรงงานรัฐวิสาหกิจ",
    "พระราชบัญญัติวินัยการเงินการคลังของรัฐ พ.ศ. 2561": "วินัยงบประมาณ, กรอบการใช้จ่ายเงินแผ่นดิน, การก่อหนี้สาธารณะ",
    "พระราชบัญญัติวิธีการงบประมาณ พ.ศ. 2561": "การจัดทำและการบริหารงบประมาณรายจ่ายประจำปี",
    "พระราชบัญญัติภาษีเงินได้ปิโตรเลียม พ.ศ. 2514": "การจัดเก็บภาษีเงินได้จากการประกอบกิจการปิโตรเลียม",
    "พระราชบัญญัติทุนรัฐวิสาหกิจ พ.ศ. 2542": "การเปลี่ยนทุนรัฐวิสาหกิจเป็นหุ้น, การแปรรูปรัฐวิสาหกิจ",
    "พระราชบัญญัติคุณสมบัติมาตรฐานสำหรับกรรมการและพนักงานรัฐวิสาหกิจ พ.ศ. 2518": "คุณสมบัติและลักษณะต้องห้ามของบุคลากรรัฐวิสาหกิจ",
    "พระราชบัญญัติว่าด้วยความผิดของพนักงานในองค์การหรือหน่วยงานของรัฐ พ.ศ. 2502": "ความผิดต่อหน้าที่ของพนักงานองค์การของรัฐ",
    "พระราชบัญญัติว่าด้วยความผิดทางวินัยของข้าราชการซึ่งไปปฏิบัติหน้าที่ในหน่วยงานที่มิใช่ส่วนราชการ พ.ศ. 2534": "วินัยข้าราชการที่ไปช่วยงานรัฐวิสาหกิจ",
    "พระราชบัญญัติว่าด้วยการจัดตั้งองค์การของรัฐบาล พ.ศ. 2496": "การจัดตั้งองค์การของรัฐบาลโดยพระราชกฤษฎีกา",
    "พระราชบัญญัติสภาพัฒนาการเศรษฐกิจและสังคมแห่งชาติ พ.ศ. 2561": "อำนาจหน้าที่ของสภาพัฒน์ (สศช.)",
    "พระราชบัญญัติการจัดการหุ้นส่วนและหุ้นของรัฐมนตรี พ.ศ. 2543": "ข้อห้ามและการโอนหุ้นของรัฐมนตรีให้นิติบุคคลจัดการ",
    "พระราชบัญญัติกำหนดความผิดเกี่ยวกับห้างหุ้นส่วนจดทะเบียน ห้างหุ้นส่วนจำกัด บริษัทจำกัด สมาคม และมูลนิธิ พ.ศ. 2499": "บทลงโทษความผิดเกี่ยวกับการบริหารนิติบุคคล",
    "พระราชบัญญัติยกเลิกพระราชบัญญัติการซื้อขายสินค้าเกษตรล่วงหน้า พ.ศ. 2542 พ.ศ. 2558": "การโอนย้ายธุรกรรมสินค้าเกษตรล่วงหน้า"
}

# Example questions
example_questions = {
    "ประมวลกฎหมายแพ่งและพาณิชย์": ["มัดจำคืออะไร", "สัญญาคืออะไร"],
    "พระราชบัญญัติการพัฒนาการกำกับดูแลและบริหารรัฐวิสาหกิจ พ.ศ. 2562": ["การประชุมของรัฐวิสาหกิจ"],
    "พระราชบัญญัติประกอบรัฐธรรมนูญว่าด้วยการป้องกันและปราบปรามการทุจริต พ.ศ. 2561": ["หน้าที่ของ ป.ป.ช.", "วิธีป้องกันการทุจริต"],
    "ประมวลรัษฎากร": ["การยื่นภาษีเงินได้บุคคลธรรมดา"],
    "พระราชบัญญัติบริษัทมหาชนจำกัด พ.ศ. 2535": ["วิธีจัดตั้งบริษัทมหาชนจำกัด"]
}

# Load models and data
def load_models_and_data(model_dir: str = "./data"):
    print("Loading models and data...", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if not os.path.exists(model_dir):
            print(f"Error: Directory '{model_dir}' not found. Please ensure your data is in this folder.", file=sys.stderr)
            return None, None, None, None, None, None, device
        
        # Load ChromaDB
        client = chromadb.PersistentClient(path=f"{model_dir}/chroma_db")
        collection = client.get_collection(name="thai_legal")
        
        # Load passages
        with open(f"{model_dir}/passages.pkl", "rb") as f:
            passages = pickle.load(f)
            
        # Load Sentence Transformer for dense retrieval
        retriever_model = SentenceTransformer("intfloat/multilingual-e5-small", device=device)
        
        # Prepare for BM25 (sparse retrieval) with PyThaiNLP word tokenization
        passage_texts = [p["text"] for p in passages]
        print("Building BM25 index with PyThaiNLP...", file=sys.stderr)
        tokenized_corpus = [word_tokenize(text.lower(), engine="newmm") for text in passage_texts]
        bm25_model = BM25Okapi(tokenized_corpus)
        
        # Load LLM (Tokenizer and LoRA) from local path first, fallback to Hugging Face Hub
        local_lora_dir = f"{model_dir}/trained_lora_model"
        if os.path.exists(local_lora_dir) and os.path.exists(f"{local_lora_dir}/adapter_config.json"):
            print(f"Loading LoRA adapter from local path: {local_lora_dir}", file=sys.stderr)
            lora_source = local_lora_dir
            llm_tokenizer = AutoTokenizer.from_pretrained(lora_source, local_files_only=True)
        else:
            lora_source = "DuckerMaster/thai-legal-lora"
            print(f"Loading LoRA adapter from Hugging Face Hub: {lora_source}", file=sys.stderr)
            llm_tokenizer = AutoTokenizer.from_pretrained(lora_source)

        llm_tokenizer.pad_token = llm_tokenizer.eos_token
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
        llm_model = PeftModel.from_pretrained(base_model, lora_source)
        print("Models and data loaded successfully!", file=sys.stderr)
        return passages, collection, retriever_model, bm25_model, llm_model, llm_tokenizer, device
    except Exception as e:
        print(f"Error loading models/data: {e}", file=sys.stderr)
        return None, None, None, None, None, None, device

# Load models globally
passages, collection, retriever_model, bm25_model, llm_model, llm_tokenizer, device = load_models_and_data()
if not all([passages, collection, retriever_model, bm25_model, llm_model, llm_tokenizer]):
    print("Application cannot run due to failed component loading. Exiting.", file=sys.stderr)
    exit(1)

# BM25 Stopwords to prevent lexical traps on procedural/question phrasing
BM25_STOPWORDS = {
    'เงื่อนไข', 'อย่างไร', 'ขั้นตอน', 'มีอะไรบ้าง', 'ข้อกำหนด', 'คืออะไร', 'กรณีใด',
    'อะไร', 'ทำไม', 'อย่างไรบ้าง', 'มีผลอย่างไร', 'ต้องทำอย่างไร', 'หรือไม', 'หรือไม่',
    'การ', 'ความ', 'มี', 'ได้', 'ต้อง', 'ให้', 'ตาม', 'ใน', 'ของ', 'และ', 'หรือ', 'ถ้า',
    'ผู้', 'ที่', 'ซึ่ง', 'อัน', 'ไว้', 'แก่', 'ต่อ', 'ด้วย', 'แห่ง', 'นั้น', 'นี้',
    'เป็นอย่างไร', 'หมายความว่าอย่างไร'
}

# Utility functions
def get_text(passage: dict) -> str:
    return passage.get('text', '') if isinstance(passage, dict) else str(passage)

def get_meta(passage: dict, key: str, default: str = '') -> str:
    return passage.get('metadata', {}).get(key, default)

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

# Retrieval
def retrieve_documents(query: str, collection, retriever_model, bm25_model, passages, device, k: int = 1):
    tokenized_query = tokenize_query_bm25(query)
    bm25_scores = np.array(bm25_model.get_scores(tokenized_query), dtype=np.float32)
    max_bm25 = float(bm25_scores.max()) if len(bm25_scores) > 0 and bm25_scores.max() > 0 else 0.0

    n_candidates = min(100, len(passages))
    top_bm25_indices = set(np.argsort(bm25_scores)[-n_candidates:])

    # Dense query with E5 query: prefix
    query_text = f"query: {query}" if not query.startswith("query: ") else query
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

    # Fast document lookup to passage index
    doc_to_idx = {p.get("text", ""): idx for idx, p in enumerate(passages)}
    candidate_indices = set(top_bm25_indices)
    for doc in dense_dict:
        if doc in doc_to_idx:
            candidate_indices.add(doc_to_idx[doc])

    results = []
    for idx in candidate_indices:
        passage = passages[idx]
        doc_text = passage.get("text", "")

        sparse_score = (bm25_scores[idx] / max_bm25) if max_bm25 > 0 else 0.0
        dense_score = dense_dict.get(doc_text, 0.0)

        # 50% Dense + 50% Sparse
        combined_score = 0.5 * dense_score + 0.5 * sparse_score
        if any(kw in doc_text for kw in LEGAL_KEYWORDS):
            combined_score += 0.05

        results.append({"passage": passage, "score": float(combined_score)})

    return sorted(results, key=lambda x: x["score"], reverse=True)[:k]

# Answer generation
def generate_llm_answer(question: str, passage: dict, score: float, model, tokenizer, device) -> str:
    if not model or not tokenizer or not passage:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"
    
    try:
        passage_text = get_text(passage).strip()
        law_title = get_meta(passage, 'law_title', '')
        section = str(get_meta(passage, 'section', ''))
        if law_title and law_title not in passage_text[:80]:
            context = f"{law_title} มาตรา {section}: {passage_text}"
        else:
            context = passage_text

        if not context.strip():
            return "ไม่พบข้อมูลที่เกี่ยวข้อง"

        prompt = f"จากกฎหมายต่อไปนี้:\n{context}\n\nคำถาม: {question}\nคำตอบ:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
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
        return f"ข้อมูลที่เกี่ยวข้อง: {get_text(passage)[:300]}..."

# Main answer function
def get_answer(question: str, history):
    start_time = time.time()
    cleaned_question = question.strip()
    
    if not cleaned_question or len(cleaned_question) < 2:
        output = f"💡 คำอธิบาย: กรุณาระบุคำถามที่ต้องการค้นหา\n⏱️ Time taken: {time.time() - start_time:.2f} seconds"
        new_history = history + [{'role': 'user', 'content': cleaned_question}, {'role': 'assistant', 'content': output}]
        return "", new_history
    
    retrieved = retrieve_documents(question, collection, retriever_model, bm25_model, passages, device, k=3)
    if not retrieved:
        output = f"💡 คำอธิบาย: ไม่พบข้อมูลที่เกี่ยวข้อง\n⏱️ Time taken: {time.time() - start_time:.2f} seconds"
        new_history = history + [{'role': 'user', 'content': cleaned_question}, {'role': 'assistant', 'content': output}]
        return "", new_history
    
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
        new_history = history + [{'role': 'user', 'content': cleaned_question}, {'role': 'assistant', 'content': output}]
        return "", new_history

    response = generate_llm_answer(question, top_passage, score, llm_model, llm_tokenizer, device)
    
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
    
    new_history = history + [{'role': 'user', 'content': cleaned_question}, {'role': 'assistant', 'content': output}]
    return "", new_history

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft(), title="บอทตอบคำถามกฎหมายไทย", css=".gr-button {margin: 5px;} .gr-panel {font-size: 14px; overflow-y: auto; max-height: 600px;}") as demo:
    gr.Markdown("""
    # 🤖 บอทตอบคำถามกฎหมายไทย (ธุรกิจและพาณิชย์)
    💡 พิมพ์คำถามเกี่ยวกับกฎหมายธุรกิจและพาณิชย์ไทย หรือเลือกจากตัวอย่างด้านล่าง
    📚 รองรับ 35 พระราชบัญญัติหลักในคลังข้อมูล (ดูรายการด้านขวา)
    🔄 ใช้ปุ่ม "ล้าง" เพื่อรีเซ็ตช่องพิมพ์ หรือ "ล้างประวัติ" เพื่อล้างประวัติการสนทนา
    💡 ตัวอย่างคำถาม: การประชุม, หน้าที่ ป.ป.ช., มัดจำ, สัญญาจัดตั้งห้างหุ้นส่วน
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(label="👤 ถามคำถามเกี่ยวกับกฎหมาย", placeholder="เช่น มัดจำคืออะไร", lines=2)
            with gr.Row():
                submit_button = gr.Button("ส่งคำถาม", variant="primary")
                clear_button = gr.Button("ล้าง", variant="secondary")
                clear_history_button = gr.Button("ล้างประวัติ", variant="secondary")
            
            gr.Markdown("## 📜 ประวัติคำถาม")
            history_output = gr.Chatbot(label="คำถามและคำตอบที่ผ่านมา", height=300, type="messages", show_label=False)
            
            gr.Markdown("## 📋 ตัวอย่างคำถาม")
            with gr.Column():
                for law, questions in example_questions.items():
                    with gr.Accordion(law, open=False):
                        for q in questions:
                            gr.Button(q, size="sm").click(
                                lambda q_text: q_text,
                                inputs=[gr.State(q)],
                                outputs=question_input,
                                queue=False,
                            ).then(
                                get_answer,
                                inputs=[question_input, history_output],
                                outputs=[question_input, history_output]
                            )
        
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("## 📚 หมวดหมู่กฎหมายที่รองรับ")
            for law, desc in LEGAL_CATEGORIES.items():
                gr.Markdown(f"• **{law}**: {desc}")

    submit_button.click(
        fn=get_answer,
        inputs=[question_input, history_output],
        outputs=[question_input, history_output]
    )
    
    clear_button.click(
        fn=lambda: "",
        inputs=None,
        outputs=question_input
    )
    
    clear_history_button.click(
        fn=lambda: None,
        inputs=None,
        outputs=history_output
    )

if __name__ == "__main__":
    demo.launch()