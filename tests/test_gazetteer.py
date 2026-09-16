"""Rules for deciding whether a gazetteer name can be looked for in running text.

Each case here is a name that produced a wrong answer in an earlier scan.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data" / "gazetteer"))

from build import acceptable, split_by_script


def test_katakana_fragments_are_rejected():
    # These were the top Japanese "place names" in a first scan, matching
    # inside loanwords.
    for fragment in ["ター", "ルア", "サル", "パラ"]:
        assert not acceptable(fragment)


def test_long_katakana_place_names_are_kept():
    assert acceptable("サンフランシスコ")
    assert acceptable("ニューヨーク")


def test_administrative_suffix_overrides_the_length_floor():
    assert acceptable("港区")
    assert acceptable("葉山町")
    assert acceptable("サバ州")


def test_short_latin_is_rejected():
    for word in ["As", "Of", "Ely", "Ash"]:
        assert not acceptable(word)


def test_scripts_are_separated_for_scanning():
    latin, cjk = split_by_script(["Tokyo", "渋谷区", "Paris", "サバ州"])
    assert latin == ["Paris", "Tokyo"]
    assert cjk == ["サバ州", "渋谷区"]
