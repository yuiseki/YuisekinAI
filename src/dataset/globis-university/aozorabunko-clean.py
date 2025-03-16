import datasets

# https://huggingface.co/datasets/globis-university/aozorabunko-clean
# License: CC BY 4.0
# Size: 382 MB

dataset = datasets.load_dataset("globis-university/aozorabunko-clean", split="train", trust_remote_code=True)
for data in dataset:
    # remove empty text
    if data["text"] == "":
      continue
    # remove text with only a newline character
    if data["text"] == "\n":
        continue
    print(data["text"])
