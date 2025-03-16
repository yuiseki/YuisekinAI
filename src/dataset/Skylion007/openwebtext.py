import datasets

# https://huggingface.co/datasets/Skylion007/openwebtext
# License: CC0-1.0
# Size: 55.21 GB

dataset = datasets.load_dataset("Skylion007/openwebtext", split="train[:20%]", trust_remote_code=True)
for data in dataset:
    # remove empty text
    if data["text"] == "":
      continue
    # remove text with only a newline character
    if data["text"] == "\n":
        continue
    print(data["text"])
