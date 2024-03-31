import datasets

wikipedia_en_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
for data in wikipedia_en_dataset.filter(lambda data, indice: indice % 50 == 0, with_indices=True):
    if data["text"] == "" or data["text"] == "\n":
        continue
    print(data["text"])

wikipedia_ja_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
for data in wikipedia_ja_dataset.filter(lambda data, indice: indice % 10 == 0, with_indices=True):
    if data["text"] == "" or data["text"] == "\n":
        continue
    print(data["text"])
