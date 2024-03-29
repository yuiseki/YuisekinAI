from transformers import LlamaTokenizer

MODEL_PREFIX = "./tmp/YuisekinAI"
OUTPUT_MODEL_DIR = "./checkpoints/YuisekinAITokenizer"

# Transformer API
tokenizer = LlamaTokenizer(
    vocab_file=MODEL_PREFIX + ".model",
    unk_token="[UNK]",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="[PAD]",
    legacy=False,
    model_max_length=50000,
)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
