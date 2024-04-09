import datasets


def remove_empty_lines(text: str) -> str:
    return "\n".join([line for line in text.splitlines() if line.strip() != ""])


wikipedia_en_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
for data in wikipedia_en_dataset.filter(lambda data, indice: indice % 5 == 0, with_indices=True):
    if data["text"] == "" or data["text"] == "\n":
        continue
    # 2つ以上連続する改行を除去する
    text = remove_empty_lines(data["text"])
    print(text)

aya_en_dataset = datasets.load_dataset("CohereForAI/aya_dataset", split="train")
for data in aya_en_dataset.filter(lambda data: data["language_code"] == "eng"):
    if data["inputs"] == "" or data["inputs"] == "\n":
        continue
    input = data["inputs"]
    # 2つ以上連続する改行を除去する
    input = remove_empty_lines(input)
    print(input)

wizardlm_en_dataset = datasets.load_dataset("WizardLM/WizardLM_evol_instruct_70k", split="train")
for data in wizardlm_en_dataset:
    # instruction
    if data["instruction"] == "" or data["instruction"] == "\n":
        continue
    # 2つ以上連続する改行を除去する
    instruction = remove_empty_lines(data["instruction"])
    # output
    if data["output"] == "" or data["output"] == "\n":
        continue
    # 2つ以上連続する改行を除去する
    output = remove_empty_lines(data["output"])
    print(output)


dolphin_en_dataset = datasets.load_dataset("cognitivecomputations/dolphin", "flan1m-alpaca-uncensored", split="train")
for data in dolphin_en_dataset:
    # instruction
    if data["instruction"] == "" or data["instruction"] == "\n":
        continue
    # 2つ以上連続する改行を除去する
    instruction = remove_empty_lines(data["instruction"])
    print(instruction)
    # input
    if data["input"] == "" or data["input"] == "\n":
        continue
    # 2つ以上連続する改行を除去する
    input = remove_empty_lines(data["input"])
    print(input)
    # output
    if data["output"] == "" or data["output"] == "\n":
        continue
    # 2つ以上連続する改行を除去する
    output = remove_empty_lines(data["output"])
    print(output)

alpaca_en_dataset = datasets.load_dataset("sahil2801/CodeAlpaca-20k", split="train")
for data in alpaca_en_dataset:
    # instruction
    if data["instruction"] == "" or data["instruction"] == "\n":
        continue
    # 2つ以上連続する改行を除去する
    instruction = remove_empty_lines(data["instruction"])
    print(instruction)
    # input
    if data["input"] == "" or data["input"] == "\n":
        continue
    # 2つ以上連続する改行を除去する
    input = remove_empty_lines(data["input"])
    print(input)
    # output
    if data["output"] == "" or data["output"] == "\n":
        continue
    # 2つ以上連続する改行を除去する
    output = remove_empty_lines(data["output"])
    print(output)


math_en_dataset = datasets.load_dataset("nvidia/OpenMathInstruct-1", split="train")
for data in math_en_dataset.filter(lambda data: data["is_correct"] is True):
    # question
    if data["question"] == "" or data["question"] == "\n":
        continue
    # 2つ以上連続する改行を除去する
    question = remove_empty_lines(data["question"])
    print(question)
    # generated_solution
    if data["generated_solution"] == "" or data["generated_solution"] == "\n":
        continue
    # 2つ以上連続する改行を除去する
    generated_solution = remove_empty_lines(data["generated_solution"])
    print(generated_solution)
