"""Stream a MediaWiki full-history XML dump and emit the latest revision of each page.

Reads the dump on stdin. Writes one file per kept namespace into --out, and a
tab-separated summary of every namespace seen to stdout.
"""

import argparse
import os
import sys
from xml.etree.ElementTree import ParseError, iterparse

MW = "{http://www.mediawiki.org/xml/export-0.11/}"

# Namespaces worth keeping as prose. Talk, User, File, Template, Module,
# Category and the wikibase Item/Property spaces are counted but not written.
KEEP = {
    0: "main_en",
    12: "help",
    212: "ja",
    3000: "proposal",
}


def iter_latest(stream):
    """Yield (namespace_id, title, wikitext) for the last revision of each page."""
    page = None
    title = None
    ns = None
    latest = None

    for event, elem in iterparse(stream, events=("start", "end")):
        if event == "start":
            if elem.tag == MW + "page":
                page = elem
                title = None
                ns = None
                latest = None
            continue

        tag = elem.tag
        if tag == MW + "title" and page is not None and title is None:
            title = elem.text
        elif tag == MW + "ns" and page is not None and ns is None:
            ns = int(elem.text) if elem.text is not None else None
        elif tag == MW + "revision" and page is not None:
            # Full-history dumps list revisions oldest first, so the last one wins.
            text = elem.find(MW + "text")
            latest = text.text if text is not None else None
            # Drop the revision so that a long history does not accumulate.
            page.remove(elem)
        elif tag == MW + "page":
            if latest:
                yield ns, title, latest
            elem.clear()
            page = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    # A truncated stream is an error unless the caller says it is sampling on
    # purpose. Swallowing it by default would make a run that died halfway
    # look like a run that finished.
    ap.add_argument("--allow-truncated", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    handles = {
        ns: open(os.path.join(args.out, name + ".txt"), "w", encoding="utf-8")
        for ns, name in KEEP.items()
    }

    counts = {}
    nbytes = {}
    truncated = False
    try:
        for ns, title, text in iter_latest(sys.stdin.buffer):
            counts[ns] = counts.get(ns, 0) + 1
            nbytes[ns] = nbytes.get(ns, 0) + len(text.encode("utf-8"))
            handle = handles.get(ns)
            if handle is not None:
                # Keep the title: it carries the tag or key being described.
                handle.write(title + "\n" + text + "\n\n")
    except ParseError:
        if not args.allow_truncated:
            raise
        truncated = True
    finally:
        for handle in handles.values():
            handle.close()

    if truncated:
        print("TRUNCATED: partial counts follow", file=sys.stderr)
    print(f"pages\t{sum(counts.values())}")
    print(f"bytes\t{sum(nbytes.values())}")
    print("ns\tkept\tpages\tbytes")
    for ns in sorted(counts, key=lambda n: -nbytes[n]):
        print(f"{ns}\t{KEEP.get(ns, '-')}\t{counts[ns]}\t{nbytes[ns]}")


if __name__ == "__main__":
    main()
