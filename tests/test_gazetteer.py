"""Rules for deciding whether a gazetteer name can be looked for in running text.

Each case here is a name that produced a wrong answer in an earlier scan.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data" / "gazetteer"))

from build import acceptable, is_mostly_lowercase, split_by_script


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


def test_place_names_are_not_mostly_lowercase():
    # Counts measured on the OSM Wiki English namespace, 2026-09-17.
    assert not is_mostly_lowercase("Berlin", 70, 1989)
    assert not is_mostly_lowercase("Germany", 165, 3936)
    assert not is_mostly_lowercase("Dresden", 94, 3440)
    # Essen is also the German verb, and still reads as a place here.
    assert not is_mostly_lowercase("Essen", 40, 693)


def test_ordinary_words_that_are_also_places_are_flagged():
    assert is_mostly_lowercase("Date", 4049, 580)
    assert is_mostly_lowercase("Forest", 2395, 810)
    assert is_mostly_lowercase("Time", 4366, 574)
    assert is_mostly_lowercase("Nice", 476, 87)


def test_rare_names_are_not_judged():
    # A ratio computed from a handful of occurrences says nothing.
    assert not is_mostly_lowercase("Obscureville", 9, 1)
