import datasets


def remove_empty_lines(text: str) -> str:
    """Remove empty lines from the given text."""

    return "\n".join([line for line in text.splitlines() if line.strip() != ""])


# https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset
# Use streaming to avoid downloading the entire dataset locally.
dataset = datasets.load_dataset(
    "common-pile/comma_v0.1_training_dataset",
    split="train",
    streaming=True,
)

for data in dataset:
    # Skip empty or whitespace-only texts
    if not data["text"] or data["text"].isspace():
        continue
    # Remove consecutive blank lines
    text = remove_empty_lines(data["text"])
    print(text)
