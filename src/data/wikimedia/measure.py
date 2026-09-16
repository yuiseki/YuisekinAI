"""Measure a MediaWiki pages-articles dump: main-namespace pages, cleaned, in tokens."""

import bz2
import os
import re
import sys
from xml.etree.ElementTree import iterparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wikitext import clean

JP = re.compile(r"[぀-ゟ゠-ヿ一-鿿]")


def iter_main_pages(stream):
    """Yield the wikitext of every namespace-0 page. pages-articles has one revision each."""
    ns_tag = None
    page = None
    title = ns = text = None
    for event, elem in iterparse(stream, events=("start", "end")):
        if ns_tag is None and elem.tag.startswith("{"):
            ns_tag = elem.tag.split("}")[0] + "}"
        if ns_tag is None:
            continue
        if event == "start":
            if elem.tag == ns_tag + "page":
                page = elem
                title = ns = text = None
            continue
        t = elem.tag
        if t == ns_tag + "title" and page is not None and title is None:
            title = elem.text
        elif t == ns_tag + "ns" and page is not None and ns is None:
            ns = int(elem.text) if elem.text is not None else None
        elif t == ns_tag + "text" and page is not None:
            text = elem.text
        elif t == ns_tag + "page":
            if ns == 0 and text:
                yield title, text
            elem.clear()
            page = None


def main(path, tokenizer):
    pages = 0
    raw_bytes = 0
    clean_bytes = 0
    tokens = 0
    jp_chars = 0
    all_chars = 0
    buf = []
    buf_bytes = 0
    FLUSH_AT = 1_000_000

    def flush():
        nonlocal tokens, jp_chars, all_chars, clean_bytes, buf_bytes
        if not buf:
            return
        buf_bytes = 0
        joined = "\n\n".join(buf)
        clean_bytes += len(joined.encode())
        tokens += len(tokenizer(joined, add_special_tokens=False)["input_ids"])
        jp_chars += len(JP.findall(joined))
        all_chars += len(joined)
        buf.clear()

    with bz2.open(path, "rb") as f:
        for title, text in iter_main_pages(f):
            pages += 1
            raw_bytes += len(text.encode())
            c = clean(text)
            buf.append(c)
            buf_bytes += len(c.encode())
            if buf_bytes >= FLUSH_AT:
                flush()
    flush()
    return dict(pages=pages, raw_bytes=raw_bytes, clean_bytes=clean_bytes,
                tokens=tokens, jp=jp_chars / max(all_chars, 1))


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    print(f"{'dump':16s} {'pages':>9s} {'wikitext':>10s} {'kept':>6s} {'tokens':>12s} {'ja':>6s}")
    grand = 0
    for path in sys.argv[1:]:
        r = main(path, tok)
        grand += r["tokens"]
        name = os.path.basename(path).split(".")[0]
        kept = 100 * r["clean_bytes"] / max(r["raw_bytes"], 1)
        print(f"{name:16s} {r['pages']:9,d} {r['raw_bytes']/1e6:9.1f}M {kept:5.0f}% "
              f"{r['tokens']/1e6:10.2f} M {100*r['jp']:5.0f}%")
    print(f"{'TOTAL':16s} {'':9s} {'':10s} {'':6s} {grand/1e6:10.2f} M")
