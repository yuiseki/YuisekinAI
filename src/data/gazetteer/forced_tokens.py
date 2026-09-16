"""Build the candidate list of place names to force into the tokenizer vocabulary.

Every name carries its source and licence, because the list is part of the
Data Information the OSAID requires, not just an input to a training script.

Tiers exist so that the cutoff can be moved without rebuilding: tier 1 is what
fits any vocabulary, tier 2 adds world cities in English, tier 3 adds their
established Japanese renderings.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import acceptable

# tier -> (source id, licence)
SOURCES = {
    "ne_country_en": (1, "Natural Earth admin-0 name_en", "public domain"),
    "ne_country_ja": (1, "Natural Earth admin-0 name_ja", "public domain"),
    "ne_admin1_en": (1, "Natural Earth admin-1 name_en", "public domain"),
    "ne_admin1_ja": (1, "Natural Earth admin-1 name_ja", "public domain"),
    "jp_pref": (1, "Geolonia japanese-addresses", "CC BY 4.0"),
    "jp_muni": (1, "Geolonia japanese-addresses", "CC BY 4.0"),
    "gn_city_en": (2, "GeoNames cities15000", "CC BY 4.0"),
    "gn_admin1_en": (2, "GeoNames admin1CodesASCII", "CC BY 4.0"),
    "wd_city_ja": (3, "Wikidata label, joined by English name", "CC0"),
    "wd_admin1_ja": (3, "Wikidata label, joined by English name", "CC0"),
}


def natural_earth(path, col, tag, out):
    import pyarrow.parquet as pq
    table = pq.read_table(path).to_pydict()
    for value in table.get(col, []):
        if value:
            out.setdefault(value.strip(), set()).add(tag)


def geolonia(path, out):
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["都道府県名"], set()).add("jp_pref")
            out.setdefault(row["市区町村名"], set()).add("jp_muni")


def geonames(path, tag, out, name_col=1, ascii_col=2):
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            for idx in (name_col, ascii_col):
                if len(parts) > idx and parts[idx]:
                    out.setdefault(parts[idx].strip(), set()).add(tag)


def wikidata_ja(path, english_names, tag, out):
    """Japanese labels for entities whose English label is already a candidate."""
    added = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                _, labels, *_ = json.loads(line)
            except Exception:
                continue
            en, ja = labels.get("en"), labels.get("ja")
            if en and ja and en in english_names:
                out.setdefault(ja.strip(), set()).add(tag)
                added += 1
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--natural-earth", default="/www/html/static/natural-earth")
    ap.add_argument("--geolonia", required=True)
    ap.add_argument("--geonames", required=True, help="directory holding the GeoNames dumps")
    ap.add_argument("--wikidata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tier", type=int, default=3)
    args = ap.parse_args()

    names = {}
    ne = args.natural_earth
    natural_earth(f"{ne}/ne_110m_admin_0_countries.parquet", "name_en", "ne_country_en", names)
    natural_earth(f"{ne}/ne_110m_admin_0_countries.parquet", "name_ja", "ne_country_ja", names)
    natural_earth(f"{ne}/ne_10m_admin_1_states_provinces.parquet", "name_en", "ne_admin1_en", names)
    natural_earth(f"{ne}/ne_10m_admin_1_states_provinces.parquet", "name_ja", "ne_admin1_ja", names)
    geolonia(args.geolonia, names)

    if args.max_tier >= 2:
        geonames(f"{args.geonames}/cities15000.txt", "gn_city_en", names)
        geonames(f"{args.geonames}/admin1CodesASCII.txt", "gn_admin1_en", names, 1, 2)

    if args.max_tier >= 3:
        english = {n for n, tags in names.items()
                   if any(t.endswith("_en") for t in tags)}
        wikidata_ja(args.wikidata, english, "wd_city_ja", names)

    rejected = {n for n in names if not acceptable(n)}
    for n in rejected:
        del names[n]

    with open(args.out, "w", encoding="utf-8") as out:
        for name in sorted(names):
            tiers = {SOURCES[t][0] for t in names[name] if t in SOURCES}
            out.write(f"{min(tiers) if tiers else 9}\t{name}\t{','.join(sorted(names[name]))}\n")

    from collections import Counter
    by_tier = Counter(min({SOURCES[t][0] for t in tags if t in SOURCES} or {9})
                      for tags in names.values())
    print(f"accepted\t{len(names)}")
    print(f"rejected\t{len(rejected)}")
    for tier in sorted(by_tier):
        print(f"tier{tier}\t{by_tier[tier]}")


if __name__ == "__main__":
    main()
