import datasets

# https://huggingface.co/datasets/wikimedia/wikipedia
# License: CC BY-SA 3.0
# Size: 71.8 GB

dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:40%]")
for data in dataset:
    # remove empty text
    if data["text"] == "":
      continue
    # remove text with only a newline character
    if data["text"] == "\n":
        continue
    print(data["text"])
