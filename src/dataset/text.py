import datasets

wikipedia_en_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
for data in wikipedia_en_dataset.filter(lambda item, idx: idx % 2 == 0, with_indices=True):
    # 空の行は除去
    # 改行のみの行は除去
    if data["text"] == "" or data["text"] == "\n":
        continue
    print(data["text"])

openmath_en_dataset = datasets.load_dataset("nvidia/OpenMathInstruct-1", split="train")
for data in openmath_en_dataset:
    if data["is_correct"] is False:
        continue
    print(data["question"])
    print(data["generated_solution"])

dolphin_en_dataset = datasets.load_dataset("cognitivecomputations/dolphin", "flan1m-alpaca-uncensored", split="train")
for data in dolphin_en_dataset:
    print(data["instruction"])
    print(data["input"])
    print(data["output"])

dolly_en_dataset = datasets.load_dataset("databricks/databricks-dolly-15k", split="train")
for data in dolly_en_dataset:
    print(data["instruction"])
    print(data["context"])
    print(data["response"])

aya_dataset = datasets.load_dataset("CohereForAI/aya_dataset", split="train")
for data in aya_dataset.filter(lambda x: x["language_code"] == "eng"):
    print(data["inputs"])
    print(data["targets"])

"""
wikipedia_ja_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
for data in wikipedia_ja_dataset:
    if data["text"] == "" or data["text"] == "\n":
        continue
    print(data["text"])
"""
