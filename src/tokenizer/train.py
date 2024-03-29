# 参考:
#  - https://zenn.dev/selllous/articles/transformers_pretrain_to_ft
import sentencepiece as spm

from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset

MODEL_PREFIX = "./tmp/YuisekinAI"

dataset_list = [
    {"id": "CohereForAI/aya_dataset", "config": None, "filter": {"field": "language_code", "value": "eng"}},
    {"id": "CohereForAI/aya_dataset", "config": None, "filter": {"field": "language_code", "value": "jpn"}},
    {"id": "wikimedia/wikipedia", "config": "20231101.en"},
    {"id": "wikimedia/wikipedia", "config": "20231101.ja"},
]


def ds_yielder():
    for dataset_data in dataset_list:
        print("start...", dataset_data)
        dataset_id = dataset_data["id"]
        dataset_config = dataset_data["config"]
        if dataset_config is not None:
            raw_dataset = load_dataset(dataset_id, dataset_config, split="train")
        else:
            raw_dataset = load_dataset(dataset_id, split="train")
        if "filter" in dataset_data:
            data_df = raw_dataset.to_pandas()
            filter_field = dataset_data["filter"]["field"]
            filter_value = dataset_data["filter"]["value"]
            data_df = data_df[data_df[filter_field] == filter_value]
            dataset = Dataset.from_pandas(data_df)
            ds = dataset
        else:
            ds = raw_dataset
        print("ds", ds)
        if "aya" in dataset_id:
            for v in ds["inputs"]:
                yield v
        else:
            counter = 0
            for v in ds:
                # Skip every 100th sentence to reduce the number of tokens
                counter += 1
                if counter % 100 != 0:
                    continue
                yield v["text"]


spm.SentencePieceTrainer.train(
    sentence_iterator=ds_yielder(),
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
