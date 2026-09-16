"""Build the place-name candidate list for the tokenizer vocabulary.

Sources and licences:
  Natural Earth admin-0 and admin-1   public domain, carries name_en and name_ja
  Geolonia japanese-addresses         CC BY 4.0
  GeoNames cities15000, admin1        CC BY 4.0

Latin and CJK names need different rules. English place names collide with
ordinary words (Reading, Mobile, Nice, Split, Date), so they are matched at
word boundaries and remain ambiguous even then. Japanese place names carry an
administrative suffix that disambiguates them, but bare katakana fragments
match inside loanwords, so they need a length floor.
"""

import csv
import json
import re

KATAKANA_ONLY = re.compile(r"^[゠-ヿー・]+$")
CJK = re.compile(r"[぀-ヿ一-鿿]")
ADMIN_SUFFIX = re.compile(r"(都|道|府|県|市|区|町|村|郡|州|省|島)$")

# Below this, a Latin string is more likely a word than a place.
MIN_LATIN = 4
# A katakana fragment shorter than this matches inside ordinary loanwords.
MIN_KATAKANA = 5


def acceptable(name):
    """Whether a gazetteer name is specific enough to look for in running text."""
    name = name.strip()
    if not name:
        return False
    if CJK.search(name):
        if ADMIN_SUFFIX.search(name):
            return True
        if KATAKANA_ONLY.match(name):
            return len(name) >= MIN_KATAKANA
        return len(name) >= 3
    return len(name) >= MIN_LATIN


def split_by_script(names):
    """Latin names are scanned with word boundaries, CJK names without."""
    latin, cjk = [], []
    for name in names:
        (cjk if CJK.search(name) else latin).append(name)
    return sorted(latin), sorted(cjk)
