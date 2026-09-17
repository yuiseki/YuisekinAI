# Datasets

Status: named, not yet built.

The corpus is being released as a dataset in its own right, ahead of any model.
The reason is practical: the binding constraint on the model is rented GPU time,
and none of it is needed to assemble, measure and document a corpus. The
hardest OSAID requirement is the Data Information, and a released dataset is
that requirement discharged rather than promised.

## Names

| Name | What |
| --- | --- |
| `YuisekinText-ja-tiny` | the openly licensed Japanese pretraining corpus |
| `YuisekinText-geo` | openly licensed text dense in place names, any language |
| `YuisekinAI-*` | models trained on them |

The two are siblings, not parent and child. `-ja-tiny` is cut by language and
size, so `-ja-small` and `-en-*` can follow it. `-geo` is cut by a property and
spans languages and sources, which is why it takes no size suffix.

The size suffix follows the tiny / small / base / large series that BERT, T5
and ViT established, rather than strict t-shirt sizing, because readers already
know it. Later corpora take the next size up; the intent is to reach something
that would support a 7 B model.

`-geo` carries no size suffix. It names a property rather than a scale, and
`YuisekinText-geo-tiny` would read badly.

### On calling it tiny

At 2.685 B Japanese tokens measured so far it is tiny, honestly. Japanese
corpora derived from Common Crawl are two orders of magnitude larger.

It is also, as far as this project has been able to establish, the largest
openly licensed Japanese corpus that exists. Common Pile is effectively
English-only; Common Corpus is multilingual but weighted to European languages.
Both facts belong on the card.

### Why not Pile

The Pile lineage was considered and set aside. `GeoPile` is taken, by a
satellite-imagery pretraining dataset from ICCV 2023. And the original Pile was
subject to DMCA takedowns over books3, which is an awkward name to inherit for
a project whose whole premise is that the data is openly licensed. Common Pile
could take it because reclaiming it was the point; here it would only invite
the comparison.

## YuisekinText-geo

The case for separating it is that the audiences differ. Someone building a
general Japanese model wants the whole corpus. Someone working on geographic
language wants the geographic part and would rather not carry 1.97 B tokens of
general Wikipedia to get it.

It is not limited to Japanese, and not limited to what is in `-ja-tiny`. The
criterion is density of place names, applied per language with a gazetteer for
that language. Sources already to hand that `-ja-tiny` does not take:

| Source | Size | Note |
| --- | --- | --- |
| OSM Wiki, English | 84.4 M tokens | dense by construction |
| OSM Wiki, DE RU ES FR IT NL namespaces | 104 MB wikitext | counted and skipped during the Japanese extraction |
| OSM Wiki Proposal namespace | 1.25 M tokens | how conventions were argued |
| Wikidata labels beyond Japanese | en 1.79 M, fr 799 k, de 692 k, nl 601 k, sv 394 k, es 375 k | CC0 |
| Natural Earth populated places | 7,342 cities, 101,526 name strings across 25 languages, no gaps | public domain |

### How much of Japanese Wikipedia is geographic, measured 2026-09-17

All 1,389,467 articles, counting distinct names from the 63,587-name candidate
list per article.

| Distinct places | Articles | % articles | Text | % bytes | Est. tokens |
| --- | --- | --- | --- | --- | --- |
| >= 1 | 1,105,850 | 79.6% | 6.13 GB | 89.6% | 1.77 B |
| >= 2 | 786,084 | 56.6% | 4.65 GB | 68.0% | 1.34 B |
| >= 3 | 509,366 | 36.7% | 3.59 GB | 52.4% | 1.03 B |
| >= 5 | 234,349 | 16.9% | 2.30 GB | 33.6% | 0.66 B |
| >= 10 | 66,154 | 4.8% | 1.05 GB | 15.4% | 0.30 B |
| >= 20 | 17,404 | 1.3% | 0.41 GB | 6.0% | 0.12 B |

Japanese Wikipedia is more geographic than expected: four articles in five name
at least one place.

At a threshold of five, Japanese Wikipedia alone contributes 0.66 B tokens to
`-geo`. Adding the English OSM Wiki at 84.4 M, the other-language OSM Wiki
namespaces, the Proposal namespace, e-Gov laws, which are dense by
construction, and the qualifying part of the Wikimedia sister projects puts
`-geo` somewhere near 1 B tokens. That stands alone.

At a threshold of ten it is 0.30 B from Wikipedia and perhaps 0.4 B in total: a
stronger claim to density, at less than half the size.

An earlier run of this measurement reported 7% of bytes at a threshold of five,
against the 33.6% here. It was measuring 170-character fragments rather than
articles, because the extraction had concatenated articles with a blank-line
separator and Wikipedia text contains blank lines of its own. A fragment rarely
holds five distinct place names. The rewrite that produced the table above
reads JSONL, one article per line, and uses `grep -F`, whose line numbers are
document ids; it takes 43 seconds against the whole corpus, where the first
attempt had not finished a fourteenth of it in eighteen minutes.

### The multilingual gazetteer

Natural Earth populated places is the spine of it: 7,342 cities named in 25
languages with no missing values, plus `ADM0NAME` and `ADM1NAME` giving
containment for every row, 228 countries and 2,527 first-level subdivisions.
Public domain, so it can be redistributed with no attribution obligation at
all, which nothing else here allows.

That makes the grouped-listing material discussed in `DATA_candidate.md`
available in 25 languages rather than one: a country, its subdivisions, and
their cities, as adjacency, in the form a transformer takes up most readily.

### Two reasons a document is in, and the record says which

Selection is not done for the user. A threshold baked in cannot be loosened
downstream, and different tasks want different densities, so every document
carries its measured density and the user cuts where they like.

But density can only be measured where there is a gazetteer for the language,
and some sources are dense by construction whatever the language: a Wikivoyage
article is a travel guide, an OSM Wiki page is documentation about mapping
places. Excluding those for want of a gazetteer would throw away the most
reliably geographic text in the collection.

So a document is in for one of two reasons, and the record says which:

| Field | Meaning |
| --- | --- |
| `inclusion` | `measured` or `source` |
| `place_count` | distinct gazetteer names found, or null when not measured |
| `gazetteer` | which gazetteer produced that count, or null |
| `lang` | language |
| `source` | e.g. `wikivoyage.en`, `osm_wiki.main_en`, `wikipedia.ja` |
| `licence` | per document, since the collection mixes CC0, public domain, CC BY 4.0 and CC BY-SA |

A user who trusts only measured density filters on `inclusion = measured` and
sets their own threshold. A user who wants all the geographic text takes
everything. Neither has to take our judgement on faith, and neither has to
guess why a document is there.

Measurable languages are those Natural Earth populated places covers, which is
25 with no gaps. Intersected with MOT's 43 that gives 15; Wikivoyage and
Wikinews overlap it further. Everything outside that set enters as `source` or
not at all.

### Gazetteers ship unmerged

The same reasoning that keeps the density threshold out of the dataset keeps
the gazetteers unmerged. Wikidata calls 台東区 `Taitō-ku` in English, OSM and
Natural Earth and GeoNames call it `Taito`, French has `arrondissement de
Taitō`. Picking one would discard how the place is actually written in the
other registers, and a downstream user cannot get that back.

Each name carries its source. Divergence between sources is information about
usage, not a defect to clean up.

### The two overlap

A place-dense Japanese document belongs in both. Anyone using `-ja-tiny` and
`-geo` together has to deduplicate by document id or count that text twice.
This goes on both cards rather than being left for a user to discover during a
training run.

To be adoptable on its own it needs three things beyond the text:

- Provenance and licence per document, so that a user can trace what they are
  mixing into their own corpus.
- A reproducible selection criterion, shipped with the gazetteer it was applied
  with. Without the 63,587-name candidate list, there is no way to say why a
  document is in.
- A frozen, citable artefact. A dataset that grows quietly cannot be cited.

## The English side is referenced, not bundled

`YuisekinText-ja-tiny` is Japanese. The English half of the training recipe is the
Common Pile, and it stays a pointer to <https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset>
rather than a copy inside this dataset.

Three reasons.

It would add bytes without adding content. The Common Pile is published and
maintained by EleutherAI; a copy of 500 GB alongside 2.685 B tokens of Japanese
adds volume to the artefact and nothing to it.

It would blur what the thing is. "The openly licensed Japanese corpus" is
citable. "The openly licensed Japanese corpus, plus a copy of someone else's
English one" is not a thing with a name.

And it would move a risk upstream that belongs upstream. The Comma model card
says plainly that "license laundering and inaccurate metadata can result in
erroneous license information in the Common Pile" and that they "cannot make a
guarantee that Comma v0.1-2T was trained exclusively on openly licensed text".
Redistributing it means making that licence assertion ourselves. Pointing at it
leaves the assertion with the people who can maintain it.

The two-stage design says the same thing in another way. English carries the
warmup and the stable phase, Japanese the decay. They are separate ingredients
used at separate times, not one mixture, so they are separate artefacts too.

## Wikivoyage and Wikinews, measured 2026-09-17

40 dumps, 1.8 GB compressed, 21 Wikivoyage and 19 Wikinews language editions.
Both are dense in place names by construction, so they enter `-geo` as
`inclusion = source` where no gazetteer exists for the language, and get a
measured density where one does.

| | Pages | Articles | Wikitext |
| --- | --- | --- | --- |
| As converted | 2,402,502 | | 8.48 GB |
| After removing redirects | | 1,918,198 | 8.43 GB |

### Redirects were being emitted as documents

`pages-articles` carries redirects as pages, and the first conversion wrote
them out. 484,304 of 2,402,502 pages, 20.2%, and English Wikivoyage is 49.5%
redirects. They are almost nothing in bytes, 0.05 GB, and a fifth of the
document count, which is the number every per-document statistic is computed
against. Fixed in `to_jsonl.py` with a test.

### Russian Wikinews is not what its size suggests

1,868,966 pages against 43,916 for English, a factor of 42. Mean article size
is normal at 3,249 bytes and there are no repeated titles, so neither of the
two anomalies seen earlier in this project explains it.

The titles do. Alongside real articles there are auto-generated scaffolding
pages: `Ожидаемые события 7 февраля 2092 года`, expected events on 7 February
2092, one per day decades into the future; `Лента новостей 27 мая 2018 года`,
a news feed page per date; bare date pages; yearly archives. Removing
redirects still leaves 1.47 million.

Excluded from `-geo` on those grounds. Not because it is large, but because
most of it is scaffolding rather than text.

### What the detector caught, and what it did not

Four statistics were collected per dump: article count, total bytes, mean
bytes, repeated titles. Only the article count flagged Russian Wikinews, and
only because it could be read against the other nineteen editions. Mean bytes
and repeated titles both looked entirely normal.

So a per-source statistic is not enough. The signal was a peer comparison, and
anything that checks a corpus for this class of problem has to compare like
with like rather than examine each source alone.

## Open

- The selection threshold for `-geo`, and whether the result is large enough to
  stand alone. Being measured.
- Whether the token counts quoted are stated against a named tokenizer. They
  are tokenizer-dependent, and the figures so far use Qwen3 as an instrument.
- The selection threshold for `-geo`, and whether one threshold suits every
  language or each needs its own.
