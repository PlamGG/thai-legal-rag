"""
Generation Quality Evaluation Script
วัดคุณภาพคำตอบที่ LLM สร้าง เทียบกับ ground truth
Metrics: ROUGE-L, Section Match Rate, Hallucination Rate
"""
import re
import random
import time
import torch
from collections import Counter
from datasets import load_dataset

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from existing codebase
from src.rag_engine import (
    load_models_and_data,
    retrieve_documents,
    generate_llm_answer,
    get_text,
    get_meta,
)

# ============================================================
# ROUGE-L (F1) — lightweight implementation, no extra deps
# ============================================================
def _lcs_length(x: list, y: list) -> int:
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]

def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = re.findall(r'[\u0E00-\u0E7F0-9a-zA-Z]+', prediction.lower())
    ref_tokens = re.findall(r'[\u0E00-\u0E7F0-9a-zA-Z]+', reference.lower())
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# ============================================================
# Section Match — ตรวจว่า LLM อ้างมาตราตรงกับ ground truth
# ============================================================
def extract_sections(text: str) -> set:
    """ดึงเลขมาตราจากข้อความ เช่น 'มาตรา 1012' -> {'1012'}"""
    pattern = r'มาตรา\s*(\d+(?:/\d+)?)'
    return set(re.findall(pattern, text))

def section_match(prediction: str, reference_sections: set) -> bool:
    """True ถ้าคำตอบ LLM อ้างมาตราอย่างน้อย 1 ตัวตรงกับ ground truth"""
    if not reference_sections:
        return True  # ถ้า ground truth ไม่มีมาตรา ไม่สามารถตรวจได้
    pred_sections = extract_sections(prediction)
    return len(pred_sections & reference_sections) > 0

# ============================================================
# Hallucination Check — ตรวจว่า LLM อ้างมาตราที่ไม่มีใน context
# ============================================================
def has_hallucinated_section(prediction: str, context: str) -> bool:
    """True ถ้า LLM อ้างมาตราที่ไม่ปรากฏในบริบทที่ให้ (hallucination)"""
    pred_sections = extract_sections(prediction)
    context_sections = extract_sections(context)
    if not pred_sections:
        return False  # ไม่ได้อ้างมาตราเลย = ไม่ hallucinate
    hallucinated = pred_sections - context_sections
    return len(hallucinated) > 0

# ============================================================
# Main Evaluation
# ============================================================
def main():
    print("=" * 60)
    print("🧪 Generation Quality Evaluation")
    print("=" * 60)

    # 1. Load models
    print("\n📦 Loading models and data...")
    passages, collection, retriever_model, bm25_model, llm_model, llm_tokenizer, device = load_models_and_data()
    if not all([passages, collection, retriever_model, bm25_model, llm_model, llm_tokenizer]):
        print("❌ Failed to load models")
        return

    # 2. Load test set
    print("📥 Loading test dataset...")
    ds = load_dataset("airesearch/WangchanX-Legal-ThaiCCL-RAG", split="test")
    
    # สุ่ม 20 ตัวอย่างที่มี positive_answer
    random.seed(42)
    samples = [r for r in ds if r.get("positive_answer", "").strip()]
    samples = random.sample(samples, min(20, len(samples)))
    print(f"📊 Evaluating {len(samples)} samples\n")

    # 3. Run evaluation
    rouge_scores = []
    section_matches = 0
    hallucinations = 0
    valid_section_checks = 0
    total_time = 0

    for i, sample in enumerate(samples, 1):
        question = sample["question"]
        ground_truth = sample["positive_answer"]
        
        # Extract ground truth sections from positive_contexts
        gt_sections = set()
        for ctx in sample.get("positive_contexts", []):
            meta = ctx.get("metadata", {})
            sec = meta.get("section", "")
            if sec:
                gt_sections.add(str(sec))
            # Also extract from context text
            gt_sections |= extract_sections(ctx.get("context", ""))

        # Run RAG pipeline
        start = time.time()
        retrieved = retrieve_documents(
            question, collection, retriever_model, bm25_model, passages, device, k=3
        )
        
        if not retrieved:
            print(f"  [{i:02d}] ⚠️ No retrieval results for: {question[:40]}...")
            rouge_scores.append(0.0)
            continue

        top = retrieved[0]
        context = f"{get_meta(top['passage'], 'law_title')} มาตรา {get_meta(top['passage'], 'section')}: {get_text(top['passage'])}"
        prediction = generate_llm_answer(
            question, context, top["passage"], top["score"],
            llm_model, llm_tokenizer, device
        )
        elapsed = time.time() - start
        total_time += elapsed

        # --- Metrics ---
        # ROUGE-L
        rl = rouge_l_f1(prediction, ground_truth)
        rouge_scores.append(rl)

        # Section Match
        if gt_sections:
            valid_section_checks += 1
            if section_match(prediction, gt_sections):
                section_matches += 1

        # Hallucination
        if has_hallucinated_section(prediction, context):
            hallucinations += 1

        status = "✅" if rl > 0.3 else "⚠️"
        print(f"  [{i:02d}] {status} ROUGE-L={rl:.3f} | Q: {question[:50]}... ({elapsed:.1f}s)")

    # 4. Summary
    print("\n" + "=" * 60)
    print("📊 Generation Quality Results")
    print("=" * 60)
    
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    sec_rate = section_matches / valid_section_checks if valid_section_checks > 0 else 0
    hall_rate = hallucinations / len(samples)
    avg_time = total_time / len(samples)

    print(f"  ROUGE-L (avg)          : {avg_rouge:.4f}")
    print(f"  Section Match Rate     : {sec_rate:.2%} ({section_matches}/{valid_section_checks})")
    print(f"  Hallucination Rate     : {hall_rate:.2%} ({hallucinations}/{len(samples)})")
    print(f"  Avg Generation Time    : {avg_time:.2f}s/question")
    print(f"  Total Samples          : {len(samples)}")
    print("=" * 60)

    # 5. Output for README
    print("\n📋 Copy-paste for README:")
    print(f"| ROUGE-L (avg) | **{avg_rouge:.2%}** |")
    print(f"| Section Match Rate | **{sec_rate:.2%}** ({section_matches}/{valid_section_checks}) |")
    print(f"| Hallucination Rate | **{hall_rate:.2%}** ({hallucinations}/{len(samples)}) |")
    print(f"| Avg Generation Time | {avg_time:.2f}s/question |")


if __name__ == "__main__":
    main()
