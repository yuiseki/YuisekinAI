import datasets

# https://huggingface.co/datasets/arxiv-community/arxiv_dataset

dataset = datasets.load_dataset("arxiv-community/arxiv_dataset", split="train")
for data in dataset:
    # remove empty text
    if data["text"] == "":
      continue
    # remove text with only a newline character
    if data["text"] == "\n":
        continue
    print(data["text"])
