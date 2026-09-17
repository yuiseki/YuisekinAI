"""Convert a Multilingual Open Text archive to JSONL, one article per line.

MOT ships Voice of America news, which is a United States government work and
therefore public domain; the corpus itself is CC BY 4.0.

Each article JSON carries the same text three times over, as `paragraphs`, as
`sentences` and as `tokens`, which is most of the archive's bulk. Only
`paragraphs` is kept. The archives also hold `audio`, `video` and `photo`
directories that are not articles.
"""

import argparse
import json
import os
import tarfile


def iter_articles(tgz_path):
    """Yield the parsed JSON of every article in the archive."""
    with tarfile.open(tgz_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) < 3 or parts[1] != "article" or not member.name.endswith(".json"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            try:
                yield json.load(handle)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # A malformed article is a fact about the archive, not
                # something to pass over in silence.
                print(f"PARSE FAILURE {member.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--source", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--licence", default="CC BY 4.0, underlying VOA text public domain")
    args = ap.parse_args()

    n = nbytes = empty = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for article in iter_articles(args.archive):
            paragraphs = article.get("paragraphs") or []
            text = "\n\n".join(p for p in paragraphs if p and p.strip())
            if not text.strip():
                empty += 1
                continue
            n += 1
            nbytes += len(text.encode("utf-8"))
            out.write(json.dumps({
                "id": f"{args.source}:{article.get('filename')}",
                "source": args.source,
                "lang": args.lang,
                "licence": args.licence,
                "title": article.get("title"),
                "url": article.get("url"),
                "published": article.get("time_published"),
                "text": text,
            }, ensure_ascii=False) + "\n")

    print(f"{args.source}\tarticles\t{n}")
    print(f"{args.source}\tbytes\t{nbytes}")
    print(f"{args.source}\tmean_bytes\t{nbytes // max(n, 1)}")
    print(f"{args.source}\tempty\t{empty}")


if __name__ == "__main__":
    main()
