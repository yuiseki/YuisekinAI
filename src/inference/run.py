import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKENIZER_NAME = "checkpoints/YuisekinAITokenizer"
BASE_DIR = "./checkpoints/"
MODEL_NAME = "YuisekinAI-mistral-0.7B"
MODEL_PATH = BASE_DIR + MODEL_NAME
torch.set_float32_matmul_precision("high")

DEVICE = "cuda"
if torch.cuda.is_available():
    print("cuda")
    DEVICE = "cuda"
else:
    print("cpu")
    DEVICE = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
).to(DEVICE)

prompt = "大規模言語モデルとは、"

inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=256,
        do_sample=True,
        early_stopping=False,
        top_p=0.95,
        top_k=20,
        temperature=0.5,
        no_repeat_ngram_size=2,
        num_beams=3,
        repetition_penalty=1.2,
    )

outputs_txt = tokenizer.decode(outputs[0])

print(outputs_txt)

model_size = sum(t.numel() for t in model.parameters())
print(f"Model size: {model_size/1000**2:.1f}M parameters")

model.push_to_hub(MODEL_NAME)
tokenizer.push_to_hub(MODEL_NAME)
