from transformers import XLNetTokenizer

MODEL_PREFIX = "./tmp/YuisekinAI-0.2"
OUTPUT_MODEL_DIR = "./checkpoints/YuisekinAIAutoTokenizer-v0.2"

tokenizer = XLNetTokenizer(
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
