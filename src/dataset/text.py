import datasets

livejupiter_dataset = datasets.load_dataset("yuiseki/open2ch-livejupiter-qa")

for i in livejupiter_dataset["train"]:
    print(i["question"])
    print(i["answer"])
