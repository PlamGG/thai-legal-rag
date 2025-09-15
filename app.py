import gradio as gr
from rag_inference import get_answer

def ask_legal(question):
    full, short = get_answer(question, answer_type="full"), get_answer(question, answer_type="short")
    return f"คำตอบสั้น: {short}\n\nคำตอบเต็ม: {full}"

iface = gr.Interface(
    fn=ask_legal,
    inputs="text",
    outputs="text",
    title="Thai Legal QA Bot",
    description="ถามคำถามเกี่ยวกับกฎหมายไทย"
)

iface.launch()
