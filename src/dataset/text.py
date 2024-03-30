import datasets

wikipedia_en_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
count = 0
for i in wikipedia_en_dataset:
    count += 1
    if count % 1000 != 0:
        continue
    print(i["text"])

wikipedia_ja_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
for i in wikipedia_ja_dataset:
    print(i["text"])

izumi_vanilla_dataset = datasets.load_dataset("izumi-lab/llm-japanese-dataset-vanilla", split="train")
for i in izumi_vanilla_dataset:
    print(i["instruction"])
    print(i["input"])
    print(i["output"])

aozorabunko_dataset = datasets.load_dataset("if001/aozorabunko-clean-sin", split="train")
for i in aozorabunko_dataset:
    print(i["text"])

oscar_ja_dataset = datasets.load_dataset("if001/oscar_2023_filtered", split="train")
for i in oscar_ja_dataset:
    print(i["text"])

databricks_dolly_ja_dataset = datasets.load_dataset("llm-jp/databricks-dolly-15k-ja", split="train")
for i in databricks_dolly_ja_dataset:
    print(i["instruction"])
    print(i["context"])
    print(i["response"])

open_math_ja_dataset = datasets.load_dataset("kunishou/OpenMathInstruct-1-1.8m-ja", split="train")
for i in open_math_ja_dataset:
    print(i["question_ja"])
    print(i["generated_solution_ja"])

amenokaku_dataset = datasets.load_dataset("kunishou/amenokaku-code-instruct", split="train")
for i in open_math_ja_dataset:
    print(i["instruction"])
    print(i["input"])
    print(i["output"])
