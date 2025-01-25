import datasets


def remove_empty_lines(text: str) -> str:
    return "\n".join([line for line in text.splitlines() if line.strip() != ""])


wikipedia_en_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
for data in wikipedia_en_dataset.filter(lambda data, indice: indice % 5 == 0, with_indices=True):
    if data["text"] == "" or data["text"] == "\n":
        continue
    # 2つ以上連続する改行を除去する
    text = remove_empty_lines(data["text"])
    print(text)
