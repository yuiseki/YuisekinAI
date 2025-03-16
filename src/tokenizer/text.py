import datasets


def remove_empty_lines(text: str) -> str:
    return "\n".join([line for line in text.splitlines() if line.strip() != ""])


wikipedia_en_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:10%]")
for data in wikipedia_en_dataset:
    # remove empty text
    if data["text"] == "" or data["text"] == "\n":
        continue
    # 2つ以上連続する改行を除去する
    text = remove_empty_lines(data["text"])
    print(text)

wikipedia_ja_dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.ja", split="train[:10%]")
for data in wikipedia_ja_dataset:
    # remove empty text
    if data["text"] == "" or data["text"] == "\n":
        continue
    # 2つ以上連続する改行を除去する
    text = remove_empty_lines(data["text"])
    print(text)
