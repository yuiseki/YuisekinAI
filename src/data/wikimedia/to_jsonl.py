"""Convert a MediaWiki pages-articles dump to JSONL, one article per line.

JSONL rather than concatenated text because every per-document step needs the
boundaries: density annotation, deduplication, and the token store's document
index. Joining articles with a blank line loses them, since wikitext contains
blank lines of its own.

Reports enough to notice a dump that is not what it looks like: article count,
size distribution, and repeated titles.
"""

import argparse
import bz2
import json
import os
import sys
from xml.etree.ElementTree import iterparse


REDIRECT_PREFIXES = ("#REDIRECT", "#redirect", "#перенаправление", "#WEITERLEITUNG",
                     "#転送", "#リダイレクト", "#REDIRECCIÓN", "#REDIRECIONAMENTO")


def is_redirect(text):
    """pages-articles carries redirects as pages; they are not documents."""
    return text.lstrip()[:20].upper().startswith(tuple(p.upper() for p in REDIRECT_PREFIXES))


def iter_main_pages(stream):
    """Yield (title, wikitext) for every namespace-0 page that is not a redirect."""
    ns_tag = None
    page = title = ns = text = None
    for event, elem in iterparse(stream, events=("start", "end")):
        if ns_tag is None and elem.tag.startswith("{"):
            ns_tag = elem.tag.split("}")[0] + "}"
        if ns_tag is None:
            continue
        if event == "start":
            if elem.tag == ns_tag + "page":
                page, title, ns, text = elem, None, None, None
            continue
        tag = elem.tag
        if tag == ns_tag + "title" and page is not None and title is None:
            title = elem.text
        elif tag == ns_tag + "ns" and page is not None and ns is None:
            ns = int(elem.text) if elem.text is not None else None
        elif tag == ns_tag + "text" and page is not None:
            text = elem.text
        elif tag == ns_tag + "page":
            if ns == 0 and text and not is_redirect(text):
                yield title, text
            elem.clear()
            page = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--source", required=True, help="e.g. wikivoyage.en")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--licence", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seen = {}
    n = nbytes = 0
    with bz2.open(args.dump, "rb") as f, open(args.out, "w", encoding="utf-8") as out:
        for title, text in iter_main_pages(f):
            seen[title] = seen.get(title, 0) + 1
            b = len(text.encode("utf-8"))
            n += 1
            nbytes += b
            out.write(json.dumps({
                "id": f"{args.source}:{title}",
                "source": args.source,
                "lang": args.lang,
                "licence": args.licence,
                "title": title,
                "text": text,
            }, ensure_ascii=False) + "\n")

    repeated = sum(1 for v in seen.values() if v > 1)
    print(f"{args.source}\tarticles\t{n}")
    print(f"{args.source}\tbytes\t{nbytes}")
    print(f"{args.source}\tmean_bytes\t{nbytes // max(n, 1)}")
    print(f"{args.source}\trepeated_titles\t{repeated}")


if __name__ == "__main__":
    main()
