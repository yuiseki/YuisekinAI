import datasets

# https://huggingface.co/datasets/cerebras/SlimPajama-627B
# License: Apache-2.0
# Size: 895 GB
# Token: 627 B

dataset = datasets.load_dataset("cerebras/SlimPajama-627B", split="train", trust_remote_code=True)
for data in dataset:
    # remove empty text
    if data["text"] == "":
      continue
    # remove text with only a newline character
    if data["text"] == "\n":
        continue
    # NOTE: Use only RedPajamaGithub and RedPajamaStackExchange
    if "redpajama_set_name" not in data["meta"]:
      continue
    if data["meta"]["redpajama_set_name"] not in ["RedPajamaGithub", "RedPajamaStackExchange"]:
      continue
    print(data["text"])
