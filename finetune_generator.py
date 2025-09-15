import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

# GPU optimization
torch.backends.cudnn.benchmark = True

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load dataset
ds = load_dataset("airesearch/WangchanX-Legal-ThaiCCL-RAG")
train_dataset = ds["train"].select(range(2000))  # ลดเพื่อ RTX 3060
test_dataset = ds["test"].select(range(200))

# Load model with 4-bit quantization
model_name = "scb10x/llama3.2-typhoon2-3b"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)

# LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)
for name, param in model.named_parameters():
    if "lora" not in name:
        param.requires_grad = False
model.print_trainable_parameters()

# Tokenize
def tokenize_function(examples):
    texts = []
    for q, positives, ans in zip(examples['question'], examples['positive_contexts'], examples['positive_answer']):
        context_str = '\n'.join([pc['context'] for pc in positives])
        text = f"จากกฎหมายต่อไปนี้:\n{context_str}\n\nคำถาม: {q}\nคำตอบ: {ans}"
        texts.append(text)
    tokenized = tokenizer(texts, truncation=True, max_length=1024, padding=True, return_tensors=None)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_train = train_dataset.map(tokenize_function, batched=True, batch_size=1000)
tokenized_test = test_dataset.map(tokenize_function, batched=True)
tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# DataLoader
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_loader = DataLoader(tokenized_train, batch_size=1, shuffle=True, collate_fn=data_collator)
test_loader = DataLoader(tokenized_test, batch_size=1, collate_fn=data_collator)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
accumulation_steps = 4

# Training
model.train()
for epoch in range(2):
    total_loss = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        try:
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            total_loss += loss.item() * accumulation_steps
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()**accumulation_steps:.4f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at batch {batch_idx}, skipping...")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            raise e
    print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")

# Test
model.eval()
total_test_loss = 0
for batch in test_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        total_test_loss += outputs.loss.item()
print(f"Test Loss: {total_test_loss / len(test_loader):.4f}")

# Save
model.save_pretrained("./data/trained_lora_model")
tokenizer.save_pretrained("./data/trained_lora_model")
print("Model saved!")