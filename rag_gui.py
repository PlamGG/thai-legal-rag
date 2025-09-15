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

# Legal categories
LEGAL_CATEGORIES = {
    "ประมวลกฎหมายแพ่งและพาณิชย์": "เกี่ยวกับสัญญา มัดจำ นิติกรรม ทรัพย์สิน",
    "ประมวลกฎหมายอาญา": "ความผิดอาญา โทษ การฟ้องร้องคดีอาญา",
    "ประมวลกฎหมายวิธีพิจารณาความแพ่ง": "ขั้นตอนการพิจารณาคดีแพ่ง",
    "ประมวลกฎหมายวิธีพิจารณาความอาญา": "ขั้นตอนการพิจารณาคดีอาญา",
    "ประมวลรัษฎากร": "ภาษีเงินได้ ภาษีมูลค่าเพิ่ม การยื่นภาษี",
    "พระราชบัญญัติบริษัทมหาชนจำกัด พ.ศ. 2535": "การจัดตั้งและบริหารบริษัทมหาชน",
    "พระราชบัญญัติหลักทรัพย์และตลาดหลักทรัพย์ พ.ศ. 2535": "การซื้อขายหลักทรัพย์ ตลาดหุ้น",
    "พระราชบัญญัติการพัฒนาการกำกับดูแลและบริหารรัฐวิสาหกิจ พ.ศ. 2562": "การบริหารรัฐวิสาหกิจ",
    "พระราชบัญญัติประกอบรัฐธรรมนูญว่าด้วยการป้องกันและปราบปรามการทุจริต พ.ศ. 2561": "หน้าที่ ป.ป.ช. การป้องกันทุจริต",
    "พระราชบัญญัติแรงงานสัมพันธ์ พ.ศ. 2518": "สิทธิและหน้าที่ของนายจ้างและลูกจ้าง",
    "พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541": "การคุ้มครองสวัสดิการแรงงาน",
    "พระราชบัญญัติการพนัน พ.ศ. 2478": "กฎหมายเกี่ยวกับการพนัน",
    "พระราชบัญญัติยาเสพติดให้โทษ พ.ศ. 2522": "การควบคุมยาเสพติด",
    "พระราชบัญญัติคนเข้าเมือง พ.ศ. 2522": "การจัดการคนต่างด้าว วีซ่า",
    "พระราชบัญญัติจดทะเบียนครอบครัว พ.ศ. 2478": "การจดทะเบียนสมรส หย่า",
    "พระราชบัญญัติลิขสิทธิ์ พ.ศ. 2537": "การคุ้มครองลิขสิทธิ์",
    "พระราชบัญญัติเครื่องหมายการค้า พ.ศ. 2534": "การจดทะเบียนเครื่องหมายการค้า",
    "พระราชบัญญัติสิทธิบัตร พ.ศ. 2522": "การคุ้มครองสิทธิบัตร",
    "พระราชบัญญัติการบัญชี พ.ศ. 2543": "การจัดทำบัญชี งบการเงิน",
    "พระราชบัญญัติสมาคม พ.ศ. 2499": "การจัดตั้งและบริหารสมาคม",
    "พระราชบัญญัติมูลนิธิ พ.ศ. 2535": "การจัดตั้งและบริหารมูลนิธิ",
    "พระราชบัญญัติพลังงานนิวเคลียร์เพื่อสันติ พ.ศ. 2559": "การควบคุมพลังงานนิวเคลียร์",
    "พระราชบัญญัติการค้าพลังงาน พ.ศ. 2550": "การค้าพลังงาน",
    "พระราชบัญญัติการผังเมือง พ.ศ. 2562": "การวางผังเมือง",
    "พระราชบัญญัติควบคุมอาคาร พ.ศ. 2522": "การก่อสร้างและควบคุมอาคาร",
    "พระราชบัญญัติส่งเสริมการอนุรักษ์พลังงาน พ.ศ. 2535": "การอนุรักษ์พลังงาน",
    "พระราชบัญญัติโรงงาน พ.ศ. 2535": "การจัดตั้งและบริหารโรงงาน",
    "พระราชบัญญัติธนาคารแห่งประเทศไทย พ.ศ. 2485": "การบริหารธนาคารแห่งประเทศไทย",
    "พระราชบัญญัติการเงินการคลังภาครัฐ พ.ศ. 2561": "การบริหารงบประมาณรัฐ",
    "พระราชบัญญัติวินัยการเงินการคลังของรัฐ พ.ศ. 2561": "วินัยการเงินของรัฐ",
    "พระราชบัญญัติหอการค้า พ.ศ. 2509": "การจัดตั้งและบริหารหอการค้า",
    "พระราชบัญญัติการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560": "การจัดซื้อจัดจ้างของรัฐ",
    "พระราชบัญญัติการพัฒนาดิจิทัลเพื่อเศรษฐกิจและสังคม พ.ศ. 2560": "การพัฒนาดิจิทัล",
    "พระราชบัญญัติการบริหารหนี้สาธารณะ พ.ศ. 2548": "การบริหารหนี้สาธารณะ"
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
        
        # Prepare for BM25 (sparse retrieval)
        passage_texts = [p["text"] for p in passages]
        tokenized_corpus = [re.findall(r'[\u0E00-\u0E7F0-9a-zA-Z]+', text.lower()) for text in passage_texts]
        bm25_model = BM25Okapi(tokenized_corpus)
        
        # Load LLM (Tokenizer and LoRA) from Hugging Face
        llm_tokenizer = AutoTokenizer.from_pretrained("DuckerMaster/thai-legal-lora")
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
        llm_model = PeftModel.from_pretrained(base_model, "DuckerMaster/thai-legal-lora")
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

# Utility functions
def get_text(passage: dict) -> str:
    return passage.get('text', '') if isinstance(passage, dict) else str(passage)

def get_meta(passage: dict, key: str, default: str = '') -> str:
    return passage.get('metadata', {}).get(key, default)

def tokenize_text(text: str) -> list[str]:
    return re.findall(r'[\u0E00-\u0E7F0-9a-zA-Z]+', str(text).lower())

def check_passage_relevance(question: str, passage_text: str, model, tokenizer, device) -> tuple[bool, bool]:
    question_lower = question.lower()
    passage_lower = passage_text.lower()
    question_keywords = tokenize_text(question)
    passage_keywords = tokenize_text(passage_text)
    matching_keywords = set(question_keywords) & set(passage_keywords)
    is_relevant = len(matching_keywords) >= 2 or any(kw in passage_lower for kw in question_lower.split() if len(kw) > 2)
    has_steps = 'วิธี' in question_lower and any(step_word in passage_lower for step_word in ['ขั้นตอน', 'กระบวนการ', 'ต้อง', 'ดำเนินการ'])
    return is_relevant, has_steps

# Retrieval
def retrieve_documents(query: str, collection, retriever_model, bm25_model, passages, device, k: int = 1):
    tokenized_query = tokenize_text(query)
    bm25_scores = bm25_model.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[-min(100, len(passages)):]
    
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
    return sorted(results, key=lambda x: x["score"], reverse=True)[:k]

# Answer generation
def generate_llm_answer(question: str, passage: dict, score: float, model, tokenizer, device) -> str:
    if not model or not tokenizer:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"
    
    is_relevant, has_steps = check_passage_relevance(question, get_text(passage), model, tokenizer, device)
    
    if score > 0.5 and is_relevant:
        context = f"{get_meta(passage, 'law_title')} มาตรา {get_meta(passage, 'section')}: {get_text(passage)}"
        if 'วิธี' in question and not has_steps:
            prompt = (
                f"จากข้อมูลกฎหมาย: {context}\n\n"
                f"คำถาม: {question}\n"
                f"ข้อมูลไม่มีขั้นตอนชัดเจน ให้อธิบายขั้นตอนทั่วไปที่เกี่ยวข้องกับคำถามโดยละเอียด ใช้ภาษาง่ายและชัดเจน"
            )
        else:
            prompt = f"จากข้อมูลกฎหมาย: {context}\n\nคำถาม: {question}\nตอบโดยสรุปและชัดเจน"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.amp.autocast('cuda') if device == "cuda" else torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        start_index = response.rfind("ตอบโดยสรุปและชัดเจน")
        if start_index != -1:
            response = response[start_index:].strip()

        if not response:
            return "ไม่พบข้อมูลที่เกี่ยวข้อง"
        
        return response
    return "ไม่พบข้อมูลที่เกี่ยวข้อง"

# Main answer function
def get_answer(question: str, history):
    start_time = time.time()
    cleaned_question = question.strip()
    
    if not cleaned_question or len(cleaned_question) < 3 or not any(kw in cleaned_question.lower() for kw in LEGAL_KEYWORDS):
        output = f"💡 คำอธิบาย: กรุณาระบุคำถามเป็นภาษาไทยหรือใช้คำศัพท์ทางกฎหมาย\n⏱️ Time taken: {time.time() - start_time:.2f} seconds"
        new_history = history + [{'role': 'user', 'content': cleaned_question}, {'role': 'assistant', 'content': output}]
        return "", new_history
    
    retrieved = retrieve_documents(question, collection, retriever_model, bm25_model, passages, device, k=1)
    if not retrieved:
        output = f"💡 คำอธิบาย: ไม่พบข้อมูลที่เกี่ยวข้อง\n⏱️ Time taken: {time.time() - start_time:.2f} seconds"
        new_history = history + [{'role': 'user', 'content': cleaned_question}, {'role': 'assistant', 'content': output}]
        return "", new_history
    
    top_passage = retrieved[0]['passage']
    score = retrieved[0]['score']
    response = generate_llm_answer(question, top_passage, score, llm_model, llm_tokenizer, device)
    
    if response == "ไม่พบข้อมูลที่เกี่ยวข้อง":
        output = f"💡 คำอธิบาย: ขออภัย ไม่พบคำตอบที่ตรงกับคำถามของคุณ\n\n"
        output += f"📋 ข้อมูลที่เกี่ยวข้อง:\n   📖 {get_meta(top_passage, 'law_title')} มาตรา {get_meta(top_passage, 'section')}\n"
        output += f"   📝 {get_text(top_passage)[:200]}...\n"
        output += f"   ⭐ คะแนนความเกี่ยวข้อง: {score:.2f}\n\n"
        output += f"💡 คำแนะนำ: ลองระบุคำถามให้ชัดเจนยิ่งขึ้น"
    else:
        output = f"💡 คำอธิบาย: {response}\n"
        output += f"📚 แหล่งที่มา:\n   1. {get_meta(top_passage, 'law_title')} มาตรา {get_meta(top_passage, 'section')} (Score: {score:.2f})\n"
    
    output += f"⏱️ Time taken: {time.time() - start_time:.2f} seconds"
    
    new_history = history + [{'role': 'user', 'content': cleaned_question}, {'role': 'assistant', 'content': output}]
    return "", new_history

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft(), title="บอทตอบคำถามกฎหมายไทย", css=".gr-button {margin: 5px;} .gr-panel {font-size: 14px; overflow-y: auto; max-height: 600px;}") as demo:
    gr.Markdown("""
    # 🤖 บอทตอบคำถามกฎหมายไทย
    💡 พิมพ์คำถามเกี่ยวกับกฎหมายไทย หรือเลือกจากตัวอย่างด้านล่าง
    📚 รองรับ 34 หมวดหมู่กฎหมาย (ดูรายการด้านขวา)
    🔄 ใช้ปุ่ม "ล้าง" เพื่อรีเซ็ตคำถาม หรือ "ล้างประวัติ" เพื่อล้างประวัติทั้งหมด
    💡 ตัวอย่างคำถาม: การประชุม, หน้าที่ ป.ป.ช., มัดจำ
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
        fn=lambda: ("", None),
        inputs=None,
        outputs=[question_input, history_output]
    )
    
    clear_history_button.click(
        fn=lambda: None,
        inputs=None,
        outputs=history_output
    )

if __name__ == "__main__":
    demo.launch()