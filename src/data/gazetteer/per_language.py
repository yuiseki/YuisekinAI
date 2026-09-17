"""Build one gazetteer per language, for measuring place density in that language.

Natural Earth populated places carries 25 language columns with no missing
values, which makes it the only source that gives the same coverage in every
language it covers. English and Japanese get more, from GeoNames, Geolonia and
Wikidata.

The consequence is that `place_count` is not comparable across languages: an
English document is scored against tens of thousands of names and a Dutch one
against seven thousand. That is why every count is recorded with the gazetteer
that produced it.
"""

import argparse
import os
import re

import pyogrio

HAN = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]")

# Two characters is a whole Han or kana name: 北京, 上海, 東京, 香港. A flat
# minimum of three drops 19% of the Chinese names and 14% of the Korean, which
# showed up as Chinese Wikivoyage scoring 33% of documents with a place against
# English Wikivoyage's 99.9%.
#
# One character is excluded for matchability, not validity. Japanese and
# Chinese have no word boundaries, so a one-character name matches inside
# everything: 津 is a real city and also appears in 津波, 沼津, 会津, 唐津,
# 興津. Natural Earth's one-character Japanese entries are four and all look
# like truncation (ホ, ポ, ワ, 単), but its Chinese list really does contain 津
# and 吴, and Korean has 36.
#
# The cost is a known undercount. Japanese municipality names survive because
# Geolonia carries the suffix, so 呉市 and 津市 are two characters; a document
# that writes 呉 or 津 bare is not counted.
MIN_LEN_LATIN = 3
MIN_LEN_CJK = 2


def min_length(name):
    return MIN_LEN_CJK if HAN.search(name) else MIN_LEN_LATIN

# Natural Earth column suffix -> language code used in the dumps.
NE_LANG = {
    "AR": "ar", "BN": "bn", "DE": "de", "EL": "el", "ES": "es", "FA": "fa",
    "FR": "fr", "HE": "he", "HI": "hi", "HU": "hu", "ID": "id", "IT": "it",
    "JA": "ja", "KO": "ko", "NL": "nl", "PL": "pl", "PT": "pt", "RU": "ru",
    "SV": "sv", "TR": "tr", "UK": "uk", "UR": "ur", "VI": "vi", "ZH": "zh",
    "EN": "en",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--populated-places", required=True)
    ap.add_argument("--admin-0", required=True,
                    help="ne_10m_admin_0_countries.shp, which carries the same 25 languages")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pyogrio.read_dataframe(args.populated_places, read_geometry=False)
    adm0 = pyogrio.read_dataframe(args.admin_0, read_geometry=False)
    os.makedirs(args.out, exist_ok=True)

    for suffix, lang in sorted(NE_LANG.items()):
        col = f"NAME_{suffix}"
        if col not in df.columns:
            continue
        names = set()
        for value in df[col].tolist():
            v = value.strip() if isinstance(value, str) else ""
            if v and len(v) >= min_length(v):
                names.add(v)
        # Country names in the same language. A news article names a country
        # far more often than it names any particular city, so leaving these
        # out of every language but English read as Ukrainian VOA mentioning a
        # place in 46% of articles.
        if col in adm0.columns:
            for value in adm0[col].tolist():
                v = value.strip() if isinstance(value, str) else ""
                if v and len(v) >= min_length(v):
                    names.add(v)

        # First-level subdivisions, English only: Natural Earth carries
        # ADM1NAME in English alone.
        if lang == "en" and "ADM1NAME" in df.columns:
            for value in df["ADM1NAME"].tolist():
                v = value.strip() if isinstance(value, str) else ""
                if v and len(v) >= min_length(v):
                    names.add(v)
        path = os.path.join(args.out, f"{lang}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(names)) + "\n")
        print(f"{lang}\t{len(names)}\tnatural_earth_populated_places")


if __name__ == "__main__":
    main()
