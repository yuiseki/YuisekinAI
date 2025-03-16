import datasets

# https://huggingface.co/datasets/deepmind/math_dataset
# License: Apache-2.0
# Size: 130.65 GB

dataset = datasets.load_dataset("deepmind/math_dataset", split="train", trust_remote_code=True)
for data in dataset:
    # remove empty text
    if data["text"] == "":
      continue
    # remove text with only a newline character
    if data["text"] == "\n":
        continue
    print(data["text"])
