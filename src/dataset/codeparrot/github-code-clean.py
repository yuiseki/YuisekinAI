import datasets

# NOTE: 巨大過ぎるので、ディスクに充分な余裕がある場合のみ実行すること

# https://huggingface.co/datasets/codeparrot/github-code
# https://huggingface.co/datasets/codeparrot/github-code-clean
# License: Apache-2.0
# Size: 32 GB?

languages = ["Python", "Javascript", "TypeScript", "Shell", "SQL", "Markdown"]
licenses = ["mit", "isc", "apache-2.0", "cc0-1.0", "bsd-2-clause", "bsd-3-clause"]
dataset = datasets.load_dataset("codeparrot/github-code-clean", split="train", languages=languages, licenses=licenses, trust_remote_code=True)
for data in dataset:
    # remove empty text
    if data["code"] == "":
      continue
    # remove text with only a newline character
    if data["code"] == "\n":
        continue
    print(data["code"])
