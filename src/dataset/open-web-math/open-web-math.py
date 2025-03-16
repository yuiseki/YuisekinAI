import datasets

# https://huggingface.co/datasets/open-web-math/open-web-math
# License: ODC-By 1.0
# Size: 27.4 GB
# Token: 14.7 B

dataset = datasets.load_dataset("open-web-math/open-web-math", split="train[:80%]", trust_remote_code=True)
for data in dataset:
    # remove empty text
    if data["text"] == "":
      continue
    # remove text with only a newline character
    if data["text"] == "\n":
        continue
    print(data["text"])
