import datasets

dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train[10%:15%]")
for data in dataset:
    # remove empty lines
    if data["text"] == "":
      continue
    # remove lines with only a newline character
    if data["text"] == "\n":
        continue
    print(data["text"])
