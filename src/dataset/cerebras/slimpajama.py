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
    # remove RedPajamaCommonCrawl
    if data["meta"]["redpajama_set_name"] == "RedPajamaCommonCrawl":
      continue
    # remove RedPajamaC4
    if data["meta"]["redpajama_set_name"] == "RedPajamaC4":
      continue
    # remove RedPajamaBook ≒ Book3
    if data["meta"]["redpajama_set_name"] == "RedPajamaBook":
      continue
    # remove RedPajamaArXiv
    if data["meta"]["redpajama_set_name"] == "RedPajamaArXiv":
      continue
    print(data["text"])
