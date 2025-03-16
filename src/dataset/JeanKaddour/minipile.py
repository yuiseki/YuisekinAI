import datasets

# https://huggingface.co/datasets/JeanKaddour/minipile
# License: MIT
# Size: 3.18 GB

dataset = datasets.load_dataset("JeanKaddour/minipile", split="train", trust_remote_code=True)
for data in dataset:
    # remove empty text
    if data["text"] == "":
      continue
    # remove text with only a newline character
    if data["text"] == "\n":
        continue
    print(data["text"])
