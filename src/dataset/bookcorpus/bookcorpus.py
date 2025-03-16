import datasets

# https://huggingface.co/datasets/bookcorpus/bookcorpus
# License: Unknown
# Size: 6.03 GB

dataset = datasets.load_dataset("bookcorpus/bookcorpus", split="train")
for data in dataset:
    # remove empty text
    if data["text"] == "":
      continue
    # remove text with only a newline character
    if data["text"] == "\n":
        continue
    print(data["text"])
