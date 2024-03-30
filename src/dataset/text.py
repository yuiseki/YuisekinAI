import datasets

wikipedia_ja_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
for i in wikipedia_ja_dataset:
    print(i["text"])
