"""The OSM Wiki keeps its tagging knowledge inside wikitables and templates.

A cleaner written for Wikipedia prose strips both and silently deletes the
content it was supposed to keep. These tests pin that behaviour down.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))

from wikitext import clean

TABLE = """{| class="wikitable" width="100%"
|- style="background-color:#F8F4C2"
!Key !! Value !! Description
|-
|rowspan="3"|railway || station || Railway stations are places where customers can access railway services.
|-
|halt || Stations without switches where only passenger trains stop are called halts.
|}"""


def test_table_cell_prose_survives():
    out = clean(TABLE)
    assert "Railway stations are places where customers can access railway services." in out
    assert "Stations without switches" in out


def test_table_syntax_is_removed():
    out = clean(TABLE)
    assert "wikitable" not in out
    assert "rowspan" not in out
    assert "{|" not in out and "|}" not in out


def test_tag_template_becomes_key_value():
    assert "railway=station" in clean("A station is {{Tag|railway|station}} here.")
    assert "opening_hours" in clean("See {{Key|opening_hours}}.")


def test_link_text_survives():
    assert "Public Transport" in clean("[[Proposed features/Public Transport|Public Transport]]")
    assert "Key:amenity" in clean("[[Key:amenity]]")


def test_japanese_prose_survives():
    src = "詳細は '''[[JA:Damaged buildings]]''' を参照。{{Tag|amenity|public_bath}} は銭湯です。"
    out = clean(src)
    assert "を参照" in out
    assert "amenity=public_bath" in out
    assert "銭湯" in out
