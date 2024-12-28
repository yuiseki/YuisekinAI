import datasets

# https://huggingface.co/datasets/Skylion007/openwebtext

dataset = datasets.load_dataset("Skylion007/openwebtext", split="train")
for data in dataset:
    # remove empty text
    if data["text"] == "":
      continue
    # remove text with only a newline character
    if data["text"] == "\n":
        continue
    print(data["text"])
