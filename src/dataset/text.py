import datasets

wikipedia_en_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
for data in wikipedia_en_dataset:
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

wizardlm_en_dataset = datasets.load_dataset("WizardLM/WizardLM_evol_instruct_70k", split="train")
for data in wizardlm_en_dataset:
    print(data["instruction"])
    print(data["output"])

dolphin_en_dataset = datasets.load_dataset("cognitivecomputations/dolphin", "flan1m-alpaca-uncensored", split="train")
for data in dolphin_en_dataset:
    print(data["instruction"])
    print(data["input"])
    print(data["output"])

"""
wikipedia_ja_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
for data in wikipedia_ja_dataset:
    if data["text"] == "" or data["text"] == "\n":
        continue
    print(data["text"])
"""
