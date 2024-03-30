import datasets

wikipedia_en_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
count = 0
for i in wikipedia_en_dataset:
    count += 1
    if count % 1000 != 0:
        continue
    if i["text"] == "" or i["text"] == "\n":
        continue
    print(i["text"])

wikipedia_ja_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
for i in wikipedia_ja_dataset:
    if i["text"] == "" or i["text"] == "\n":
        continue
    print(i["text"])

aozorabunko_dataset = datasets.load_dataset("if001/aozorabunko-clean-sin", split="train")
for i in aozorabunko_dataset:
    try:
        if i["text"] == "" or i["text"] == "\n":
            continue
        print(i["text"])
    except Exception:
        continue

oscar_ja_dataset = datasets.load_dataset("if001/oscar_2023_filtered", split="train")
for i in oscar_ja_dataset:
    if i["text"] == "" or i["text"] == "\n":
        continue
    print(i["text"])

izumi_vanilla_dataset = datasets.load_dataset("izumi-lab/llm-japanese-dataset-vanilla", split="train")
for i in izumi_vanilla_dataset:
    if i["instruction"] != "" and i["instruction"] != "\n":
        print(i["instruction"])
    if i["input"] != "" and i["input"] != "\n":
        print(i["input"])
    if i["output"] != "" and i["output"] != "\n":
        print(i["output"])
