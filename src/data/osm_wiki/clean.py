"""Flatten OSM Wiki wikitext to prose.

OSM Wiki differs from Wikipedia: most of its tagging knowledge sits inside
wikitables and inside {{Tag}}/{{Key}} templates rather than in paragraphs. A
cleaner that strips tables and templates deletes the content it was meant to
keep, so both are unwrapped rather than removed.
"""

import re

# {{Tag|railway|station}} -> railway=station, {{Key|railway}} -> railway
TAG_TPL = re.compile(r"\{\{\s*(?:Tag|Key|Value|IconNode)\s*\|([^{}|]*)(?:\|([^{}|=]*))?(?:\|[^{}]*)?\}\}", re.I)


def _tag_repl(m):
    key = (m.group(1) or "").strip()
    val = (m.group(2) or "").strip()
    if key and val:
        return f"{key}={val}"
    return key


def clean(t):
    # Unwrap tag templates first, innermost-out, before anything strips braces.
    for _ in range(3):
        new = TAG_TPL.sub(_tag_repl, t)
        if new == t:
            break
        t = new

    # Remaining templates: keep argument values that read like prose, drop the rest.
    def tpl_repl(m):
        body = m.group(1)
        parts = [p.strip() for p in body.split("|")[1:]]
        keep = []
        for p in parts:
            if "=" in p:
                p = p.split("=", 1)[1].strip()
            if len(p) > 25 and " " in p:
                keep.append(p)
        return " " + " ".join(keep) + " "

    for _ in range(4):
        new = re.sub(r"\{\{([^{}]*)\}\}", tpl_repl, t)
        if new == t:
            break
        t = new

    t = re.sub(r"<ref[^>]*>.*?</ref>", " ", t, flags=re.S)
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)

    # Tables: drop the syntax, keep the cells. This is where the content is.
    out = []
    for line in t.split("\n"):
        s = line.strip()
        if s.startswith("{|") or s.startswith("|}") or s.startswith("|-"):
            continue
        if s.startswith("|") or s.startswith("!"):
            s = s.lstrip("|!")
            # A leading "class=..." style attribute block ends at the first |
            cells = re.split(r"\|\||!!", s)
            cleaned = []
            for c in cells:
                c = c.strip()
                # Drop cell attributes such as rowspan="3" or style="..."
                c = re.sub(r'^\s*(?:[a-z-]+\s*=\s*"[^"]*"\s*)+\|', "", c)
                c = re.sub(r'^\s*(?:[a-z-]+\s*=\s*[\w#%]+\s*)+\|', "", c)
                if c:
                    cleaned.append(c)
            s = " | ".join(cleaned)
        out.append(s)
    t = "\n".join(out)

    t = re.sub(r"\[\[(?:[^\[\]|]*\|)?([^\[\]|]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\]", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"'''?", "", t)
    t = re.sub(r"^=+\s*(.*?)\s*=+$", r"\1", t, flags=re.M)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()
