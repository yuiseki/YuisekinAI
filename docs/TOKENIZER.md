# Tokenizer

Status: experiment in progress. Nothing here is decided.

The vocabulary is the one thing pretraining fixes that cannot be changed
afterwards. Post-training sees the world through whatever vocabulary it is
given, and the place hierarchy is a post-training concern (see
`DATA_candidate.md`), so the vocabulary has to be chosen with that in mind now.

## The idea

Make Japanese and world place names single tokens, and publish the tokenizer as
an artefact in its own right. It is small, it runs on the SentencePiece Lite
runtime in about 45 KB with no dependencies, and no openly licensed
geography-aware Japanese tokenizer appears to exist.

## What is affordable, by hierarchy level

Measured from Geolonia's japanese-addresses, 2026-09-17.

| Level | Unique | Share of a 64k vocabulary |
| --- | --- | --- |
| Prefectures | 47 | 0.07% |
| Municipalities | 1,892 | 3.0% |
| Oaza/chome | 147,385 | does not fit |
| Oaza base, chome suffix removed | 88,472 | does not fit |

Prefecture and municipality together are 1,939 strings and cost 3% of a 64k
vocabulary. That is cheap, and the saving is large: municipality names have a
median length of 6 characters, so forcing one to a single token saves around
five tokens every time it appears.

Below that level it inverts. Oaza base names have a median length of 4
characters, so the saving is two or three tokens, bought with 88,472 slots.
They are also compositional: chome numbering is regular, and the base name
segments naturally.

## World coverage

| Source | Unique names | Licence |
| --- | --- | --- |
| Natural Earth admin-0, name_en and name_ja | ~350 | public domain |
| Natural Earth admin-1, name_en and name_ja | ~9,000 | public domain |
| Geolonia, prefectures and municipalities | 1,939 | CC BY 4.0 |
| GeoNames admin1 | 4,323 | CC BY 4.0 |
| GeoNames cities15000, name and asciiname | 38,769 | CC BY 4.0 |
| GeoNames admin2 | 57,041 | CC BY 4.0 |
| GeoNames cities5000 | 77,667 | CC BY 4.0 |
| Wikidata labels, Japanese | 223,550 distinct | CC0 |

Natural Earth carries `name_ja` alongside `name_en` for both countries and
first-level subdivisions, which is unusual and useful: multilingual place names
in the public domain, with no attribution obligation at all.

### Japanese names for places outside Japan

Natural Earth carries `name_ja` for countries and first-level subdivisions but
stops there, and the corpus measurement below found Japanese renderings of
foreign places almost absent: 487 of 4,472 admin-1 names.

Wikidata fills it, and is CC0. A local extract of Wikidata labels for
OSM-linked entities (`/www/html/static/openstreetmap/names/wikidata_names.json`,
482 MB, 2,198,819 entities) carries 242,929 Japanese labels, 223,550 of them
distinct, and 191,598 English-to-Japanese pairs. Joined to the gazetteer by
English name:

| Target | Names | With a Japanese label |
| --- | --- | --- |
| GeoNames cities15000 | 32,128 | 14,864 (46%) |
| GeoNames admin1 | 3,784 | 1,574 (42%) |

These are established Japanese renderings, 四川省 and ヘルダーラント州 and
ギュミュシュハーネ県, not transliterations produced on the fly.

The file covers every kind of OSM-linked feature, not only administrative
units, so it holds universities, stations and mountains alongside places and
needs filtering through the gazetteer rather than being used whole.

### Tiers

| Tier | Contents | Tokens | Of 64k | Of 128k |
| --- | --- | --- | --- | --- |
| 1 | Natural Earth plus Japanese prefectures and municipalities | ~11,200 | 18% | 9% |
| 1+2 | plus GeoNames cities15000, English | ~50,000 | 78% | 39% |
| 1+2+3 | plus their Japanese labels from Wikidata | ~66,400 | does not fit | 52% |

Covering the world in both languages does not fit in 64k. It is a 96k or 128k
decision, and the cost of that is in the embedding table, below.

## The cost is embeddings, not disk

An earlier note in `DESIGN.md` framed a larger vocabulary as costing disk: a
`uint32` token store instead of `uint16`, 200 GB instead of 100 GB for 50 B
tokens, against 905 GB free. That is true and affordable, and it is not the
real cost.

The real cost is the embedding table. At a hidden size of 1024, with embeddings
tied:

| Vocabulary | Embedding parameters |
| --- | --- |
| 64k | 65.5 M |
| 96k | 98 M |
| 128k | 131 M |

For a 0.5 B model a 128k vocabulary puts 26% of the parameters into the
embedding table. Small models pay for large vocabularies in a way large ones do
not.

## Matching place names is not trivial

Establishing whether a name occurs in a corpus at all turned out to need three
passes. Recording the failures because each one produced a plausible-looking
wrong answer.

A first scan reported "Date" as one of the most frequent place names in the OSM
Wiki, at 10,828 occurrences, along with South, North and West. All four are
GeoNames city names and all four were matching the ordinary English words. On
the Japanese side the top hits included the katakana fragments ター, ルア, サル
and パラ, matching inside loanwords, because the minimum candidate length had
been set at two characters for CJK.

The second scan required whole-word matching for Latin names and dropped bare
katakana below five characters unless it ended in an administrative suffix.
Japanese place names turn out to be the easy case: 渋谷区 is essentially never
not a place, because the suffix carries the disambiguation. English place names
are the hard case, because Reading, Mobile, Nice, Split and Date are ordinary
words that happen to be cities. Latin and CJK need different selection rules.

## Corpus coverage, measured 2026-09-17

Against OSM Wiki (English and Japanese) and e-Gov laws only, roughly 0.5 B
tokens, which is a small fraction of the eventual corpus.

| Group | Candidates | >=1 | >=10 | >=50 |
| --- | --- | --- | --- | --- |
| Japanese prefectures | 47 | 47 | 47 | 47 |
| Japanese municipalities | 1,892 | 1,891 | 1,015 | 256 |
| Country names, English | 177 | 177 | 175 | 167 |
| Country names, Japanese | 177 | 101 | 93 | 56 |
| Admin-1 names, English | 4,353 | 3,324 | 1,859 | 839 |
| Admin-1 names, Japanese | 4,472 | 487 | 86 | 49 |
| GeoNames cities | 38,769 | 20,134 | 8,063 | 2,389 |

The OSM Wiki covers world place names in English better than expected: 20,134
of 38,769 GeoNames city names occur in it. Japanese renderings of foreign
places do not: 487 of 4,472 admin-1 names and 101 of 177 country names. That
gap is a consequence of having measured only 1.28 M tokens of Japanese OSM
Wiki, and Wikipedia ja at 1.97 B tokens should close it.

Japanese municipality coverage comes mostly from enumeration rather than use.
The occurrence counts in the Japanese OSM Wiki cluster sharply at exactly two
or three, 1,489 of 1,892 municipalities, and a page titled `=== 01: 北海道 ===`
carries 191 municipality names in a single table. Only 70 municipalities occur
ten or more times.

This was first written down as a weakness. It is not one, for two separate
reasons.

For the tokenizer, a name listed once still earns its slot: it is a single
token every time it appears afterwards, including in input the model has never
seen. Occurrence in a shipped gazetteer, not distributed usage in the corpus,
is the criterion.

For the model, a list grouped by parent is not weak signal but clean signal. A
page that reads `=== 01: 北海道 ===` followed by 191 municipality names encodes
containment as adjacency, and adjacency is what a transformer picks up most
readily. 札幌市 and 函館市 appear in the context of 北海道 with none of the
ambiguity that scattered prose mentions carry. See the hierarchy section of
`DATA_candidate.md`, which this finding revises.

## Forcing place names: measured 2026-09-17

Two SentencePiece unigram models, 16k vocabulary, byte fallback, trained on the
same 45 MB of Japanese (30 MB of e-Gov law text plus the Japanese OSM Wiki).
One had the 1,939 prefecture and municipality names as `user_defined_symbols`,
the other had nothing forced.

Tested on nine strings chosen because a municipality name spans a morpheme
boundary inside them.

| Input | Correct | With forcing | Control |
| --- | --- | --- | --- |
| 東京都市計画 | 東京 / 都市計画 | 東京都 / 市 / 計画 (wrong) | 東京 / 都市計画 |
| 中央区分 | 中央 / 区分 | 中央区 / 分 (wrong) | 中央 / 区分 |
| 名古屋市場 | 名古屋 / 市場 | 名 / 古 / 屋 / 市場 (wrong) | 名古屋市 / 場 (wrong) |
| 大阪市場 | 大阪 / 市場 | 大阪 / 市場 | 大阪 / 市場 |
| 空港区域 | 空港 / 区域 | 空港 / 区域 | 空港 / 区域 |
| 温泉区画 | 温泉 / 区画 | 温泉 / 区画 | 温泉 / 区画 |
| 北海道内 | 北海道 / 内 | 北海道 / 内 | 北海道 / 内 |

Three wrong with forcing, one without.

The mechanism is not what it first looked like. `user_defined_symbols` are not
matched greedily ahead of segmentation: 大阪市場 stays 大阪 / 市場 even with
大阪市 forced, so the segmenter is still choosing by score. What forcing does
is distort the scores. A forced piece takes probability mass during EM, and
competing segmentations weaken. 名古屋市場 shows this most clearly: with 名古屋市
forced, 名古屋 was never learned as a piece at all and the string fell apart
into single characters, while the control had learned 名古屋市 on its own.

Which points at the real finding. Of the 1,939 names, the control model learned
49 as single tokens unaided, almost all of them prefectures. The rest cost a
median of five tokens each, 9,535 in total to spell the whole list.

So the population splits in two, and the two halves want opposite treatment.

The frequent names, prefectures and large cities, are learned without help and
are exactly the ones that collide with ordinary words: 東京都 inside
東京都市計画, 中央区 inside 中央区分. Forcing them buys nothing and breaks
things.

The long tail, あきる野市 and うきは市 and いちき串木野市, is never learned, costs
five tokens every time it appears, and collides with nothing. Forcing it is
free and is where the whole value of the idea sits.

The selection rule follows: force a name only if the tokenizer does not learn
it unaided and it does not change the segmentation of text that does not
contain it as a place. Both halves of that are testable by training twice and
diffing, which is what this experiment now is.

Caveats. The vocabulary here is 16k against a target of 64k or more, so more
names would be learned unaided at the real size. The training sample is
law-heavy, which inflates 東京都 and 中央区 specifically. Nine adversarial
strings is not a measurement of the rate on ordinary text. All three of those
need redoing at the real vocabulary size on a balanced sample.

## Open

- Vocabulary size, against measured compression on held-out Japanese prose,
  Japanese place names, world place names, OSM tag keys, law text and English
  prose.
- Whether tier 2 earns its 39,000 slots.
- The mis-segmentation rate on ordinary Japanese rather than on nine strings
  chosen to break it, at the real vocabulary size and on a balanced sample.
- Unigram or BPE.
- Whether the published tokenizer and the one this model uses are the same
  artefact.
