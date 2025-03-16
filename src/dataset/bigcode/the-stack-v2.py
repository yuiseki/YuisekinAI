import datasets

# https://huggingface.co/datasets/bigcode/the-stack-v2
# License: Mixed
# Size: 1.68 TB

dataset = datasets.load_dataset("bigcode/the-stack-v2", split="train", trust_remote_code=True)
for data in dataset:
    # remove empty text
    if data["content"] == "":
      continue
    # remove text with only a newline character
    if data["content"] == "\n":
        continue
    print(data["text"])
