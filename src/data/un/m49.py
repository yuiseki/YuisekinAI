"""Extract the UN M49 country and area names in all six UN languages.

The UNSD methodology page renders its tables into the HTML rather than serving
them from an API, and there is no CSV download. All six languages sit in the
same document, each behind an anchor: ARB, CHN, ENG, ESP, FRA, RUS. One fetch
gets them all.

Each row carries the whole containment chain, which is what makes this useful
beyond the country names: world, region, sub-region, intermediate region and
country, each with its M49 code, in six languages.
"""

import argparse
import html
import json
import re
from collections import defaultdict

SOURCE = "https://unstats.un.org/unsd/methodology/m49/overview"

ANCHORS = {"ARB": "ar", "CHN": "zh", "ENG": "en", "ESP": "es", "FRA": "fr", "RUS": "ru"}


def cell_text(cell):
    return html.unescape(re.sub(r"<[^>]+>", "", cell)).strip()


def parse(document):
    """Return {language: [row, ...]} where a row is the cells of one table line."""
    rows_by_language = defaultdict(list)
    for anchor, lang in ANCHORS.items():
        block = re.search(
            rf'id="{anchor}_Overview"(.*?)(?=id="(?:{"|".join(ANCHORS)})_Overview"|\Z)',
            document, re.S)
        if block is None:
            # A missing language is a change in the page, not something to
            # pass over: the caller needs to know the extraction is partial.
            print(f"BLOCK NOT FOUND: {anchor}")
            continue
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", block.group(1), re.S):
            cells = [cell_text(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
            cells = [c for c in cells if c]
            if len(cells) >= 8 and cells[0].isdigit():
                rows_by_language[lang].append(cells)
    return rows_by_language


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html_file")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = parse(open(args.html_file, encoding="utf-8", errors="ignore").read())
    json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    counts = {lang: len(v) for lang, v in rows.items()}
    print(f"languages\t{len(rows)}")
    for lang in sorted(counts):
        print(f"{lang}\t{counts[lang]}")
    if len(set(counts.values())) > 1:
        # The six tables describe the same list; unequal lengths mean the
        # extraction lost rows from at least one of them.
        print("ROW COUNTS DISAGREE: extraction is not consistent across languages")


if __name__ == "__main__":
    main()
