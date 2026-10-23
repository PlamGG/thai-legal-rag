import pickle
import chromadb
import torch
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from peft import PeftModel
from functools import lru_cache
from rank_bm25 import BM25Okapi

# --------- Device & Memory ----------
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------- Load passages & Chroma ----------
with open("./data/passages.pkl", "rb") as f:
    passages = pickle.load(f)
print(f"📂 Loaded {len(passages)} passages")

client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_collection(name="thai_legal")

# --------- Tokenizer & Retriever ----------
try:
    from pythainlp.tokenize import word_tokenize
    def tokenize(text):
        return word_tokenize(text, engine="newmm") + re.findall(r'[0-9a-zA-Z]+', text.lower())
except ImportError:
    def tokenize(text):
        return re.findall(r'[\u0E00-\u0E7F0-9a-zA-Z]+', text.lower())

# SentenceTransformer retriever
retriever_model = SentenceTransformer("intfloat/multilingual-e5-small", device=device)

# BM25
tokenized_corpus = [tokenize(p["text"]) for p in passages]
bm25 = BM25Okapi(tokenized_corpus)

# Dense + Sparse combined score
def calc_similarity(query, text):
    q_words, t_words = tokenize(query), tokenize(text)
    legal_keywords = ['ภาษี','คืนภาษี','ยื่นคำร้อง','ผู้ถือหุ้น','องค์ประชุม','กรรมการ','หุ้น','บริษัท']
    score = len(set(q_words) & set(t_words)) / len(q_words) if q_words else 0.0
    if any(kw in text for kw in legal_keywords): score += 0.1
    return min(score,1.0)

# --------- Model Load ----------
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
)

try:
    tokenizer = AutoTokenizer.from_pretrained("./data/trained_lora_model", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        "scb10x/llama3.2-typhoon2-3b",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config
    )
    model = PeftModel.from_pretrained(base_model, "./data/trained_lora_model")
    print("✅ Model loaded successfully")
except Exception as e:
    model, tokenizer = None, None
    print(f"❌ Model load failed: {e}")

# --------- Retrieval ----------
text_to_idx = {p["text"]: i for i, p in enumerate(passages)}

@lru_cache(maxsize=1000)
def retrieve(query, k=10):
    try:
        query_emb = retriever_model.encode([query], convert_to_numpy=True, device=device).astype('float32')
        chroma_results = collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=k*2,
            include=["documents","metadatas","distances"]
        )
        tokenized_query = tokenize(query)
        bm25_scores = bm25.get_scores(tokenized_query)
        results = []
        for doc, meta, dist in zip(
            chroma_results["documents"][0],
            chroma_results["metadatas"][0],
            chroma_results["distances"][0]
        ):
            idx = text_to_idx.get(doc, None)
            if idx is None: continue
            dense_score = 1 - dist/10
            sparse_score = bm25_scores[idx]/(max(bm25_scores)+1e-6)
            combined_score = 0.6*dense_score + 0.3*sparse_score + 0.1*calc_similarity(query, doc)
            results.append({"passage": passages[idx], "score": combined_score})
        return [x["passage"] for x in sorted(results, key=lambda x: x["score"], reverse=True)[:k]]
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        return []

# Filter relevant passages
def filter_relevant(query, passages, threshold=0.1):
    relevant = [p for p in passages if calc_similarity(query, p["text"])>=threshold]
    return relevant if relevant else passages[:1]

# --------- LLM generate with fallback ----------
def generate_llm_answer(question, context):
    if not model or not tokenizer: return None
    try:
        prompt = f"จากข้อมูลกฎหมาย:\n{context}\n\nคำถาม: {question}\nคำตอบ: ตอบสั้น 1-2 ประโยคแล้วอธิบาย"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt,"").strip()
        torch.cuda.empty_cache()
        return resp if calc_similarity(question, resp)>=0.1 else None
    except:
        return None

# Simple answer fallback
def simple_answer(passage, question):
    text = passage["text"]
    sentences = re.split(r'[.!?。]', text)
    relevant = next((s.strip() for s in sentences if calc_similarity(question,s)>=0.1), None)
    short_ans = relevant if relevant else text[:150]+"..." if len(text)>150 else text
    law = passage["metadata"].get("law_title","")
    section = passage["metadata"].get("section","")
    return f"คำตอบสั้น: {short_ans}\nคำอธิบาย: จาก {law} มาตรา {section}: {text}"

# --------- Get answer ----------
def get_answer(question, answer_type="full"):
    legal_keywords = ['บริษัท','หุ้น','กรรมการ','กฎหมาย','มาตรา','พระราชบัญญัติ','จดทะเบียน','ประชุม','ภาษี','คืนภาษี','ยื่นคำร้อง']
    if not any(kw in question.lower() for kw in legal_keywords):
        return "ขออภัย คำถามนี้ไม่เกี่ยวข้องกับกฎหมายที่มีในระบบ", []

    retrieved = retrieve(question)
    if not retrieved: return "ไม่พบข้อมูลที่เกี่ยวข้อง", []

    relevant = filter_relevant(question, retrieved)
    context = "\n".join([f"{p['metadata'].get('law_title','')} มาตรา {p['metadata'].get('section','')}: {p['text']}" for p in relevant[:3]])
    llm_resp = generate_llm_answer(question, context)

    if llm_resp:
        if answer_type=="short": return llm_resp.split("คำอธิบาย:")[0].strip(), relevant
        return llm_resp, relevant

    # Fallback simple answer
    fallback = simple_answer(relevant[0], question)
    if answer_type=="short": return fallback.split("คำอธิบาย:")[0].strip(), relevant[:1]
    return fallback, relevant[:1]

# --------- Test / Chat ----------
def run_test(questions):
    for i,q in enumerate(questions,1):
        print(f"\n❓ คำถาม {i}: {q}")
        resp,sources = get_answer(q)
        print(f"💡 {resp}")
        if sources: 
            print("📚 แหล่งที่มา:")
            for j,s in enumerate(sources[:2],1):
                print(f"   {j}. {s['metadata'].get('law_title','')} มาตรา {s['metadata'].get('section','')}")
        print("---")

def chat():
    print("🤖 Chat bot กฎหมาย")
    while True:
        q = input("ถามกฎหมาย: ").strip()
        if q.lower() in ["exit","ออก","quit"]: break
        resp,_ = get_answer(q, answer_type="short")
        print(f"🤖 {resp}")

# --------- Run ----------
if __name__=="__main__":
    test_qs = [
        "การยื่นคำร้องขอคืนภาษีต้องทำภายในกี่ปี",
        "บริษัทมหาชนจำกัดต้องมีผู้ถือหุ้นกี่คน",
        "กรรมการบริษัทมีกี่คน",
        "การประชุมผู้ถือหุ้นต้องมีองค์ประชุมกี่เปอร์เซ็นต์",
        "ผู้ถือหุ้นมีสิทธิอะไรบ้าง",
        "การออกหุ้นใหม่ต้องทำอย่างไร"
    ]
    run_test(test_qs)
    chat()
