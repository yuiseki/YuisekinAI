import datasets

# https://huggingface.co/datasets/manu/project_gutenberg
# License: https://www.gutenberg.org/policy/license.html
# Size: 14.4 GB

dataset = datasets.load_dataset("manu/project_gutenberg", split="en", trust_remote_code=True)
for data in dataset:
    # remove empty text
    if data["text"] == "":
      continue
    # remove text with only a newline character
    if data["text"] == "\n":
        continue
    # TODO:
    # All examples correspond to a single book, and contain a header and a footer of a few lines (delimited by a *** Start of *** and *** End of *** tags).
    # We remove these lines.
    print(data["text"])
