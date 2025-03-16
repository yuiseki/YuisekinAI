import datasets

# https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences
# License: CC BY-SA 4.0
# Size: 22.13 GB

dataset = datasets.load_dataset("HuggingFaceH4/stack-exchange-preferences", split="train[:10%]", trust_remote_code=True)
for data in dataset:
    # remove empty text
    if data["question"] == "":
      continue
    # remove text with only a newline character
    if data["question"] == "\n":
        continue
    text = "Question: " + data["question"]
    if not "answers" in data:
        continue
    if not data["answers"]:
        continue
    if not len(data["answers"]) > 0:
        continue
    if not "text" in data["answers"][0]:
        continue
    if not data["answers"][0]["text"]:
        continue
    answer = data["answers"][0]["text"]
    text += "\nAnswer: " + answer
    print(text)
