import datasets

wikipedia_en_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
for data in wikipedia_en_dataset.filter(lambda data, indice: indice % 100 == 0, with_indices=True):
    if data["text"] == "" or data["text"] == "\n":
        continue
    print(data["text"])

wikipedia_ja_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
for data in wikipedia_ja_dataset.filter(lambda data, indice: indice % 10 == 0, with_indices=True):
    if data["text"] == "" or data["text"] == "\n":
        continue
    print(data["text"])

oscar_ja_dataset = datasets.load_dataset("if001/oscar_2023_filtered", split="train")
for data in oscar_ja_dataset.filter(lambda data, indice: indice % 100 == 0, with_indices=True):
    if data["text"] == "" or data["text"] == "\n":
        continue
    print(data["text"])
