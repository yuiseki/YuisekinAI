# 参考:
#  - https://zenn.dev/selllous/articles/transformers_pretrain_to_ft
import sentencepiece as spm
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset

MODEL_PREFIX = "./tmp/YuisekinAI-0.2"


spm.SentencePieceTrainer.train(
    input="./tmp/tokenizer_train.txt",
    model_type="unigram",
    model_prefix=MODEL_PREFIX,  # 出力されるモデルのファイル名に使われる
    add_dummy_prefix=False,  # rinna-3.6bに習って、文章の先頭にスペースが追加されないように
    byte_fallback=True,  # rinna-3.6bに習って、未知語をutf-8バイトに分解するために
    vocab_size=50000,  # vocab number
    character_coverage=0.9995,
    unk_piece="[UNK]",
    pad_piece="[PAD]",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    train_extremely_large_corpus=True,
    # refer:https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part7.html
    input_sentence_size=120000000,
)

sp = spm.SentencePieceProcessor()
sp.Load(MODEL_PREFIX + ".model")


def tokenize(raw_text):
    tokenized = sp.encode_as_pieces(raw_text)
    return tokenized


# encode: text => is
print(sp.encode_as_pieces("これは、テストです。"))
print(sp.encode_as_ids("これは、テストです。"))

# decode: id => text
print(sp.decode_pieces(["▁", "これは", "、", "テスト", "です", "。"]))
print(sp.decode_ids([381, 260, 1662, 279, 261]))

# check vocab size
print(sp.get_piece_size())
