from transformers import LlamaTokenizer

MODEL_PREFIX = "./tmp/YuisekinAI-en"
OUTPUT_MODEL_DIR = "./checkpoints/YuisekinAILlamaTokenizer-en-v0.2"

# Transformer API
tokenizer = LlamaTokenizer(
    vocab_file=MODEL_PREFIX + ".model",
    unk_token="[UNK]",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="[PAD]",
    extra_ids=0,
    model_max_length=50000,
    add_prefix_space=False,
    legacy=False,
)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
