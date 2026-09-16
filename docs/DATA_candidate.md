# Data candidates

Sources considered for the corpus but not in it. Each entry says what it is,
what it would add, and what stands between it and `DATA.md`.

Nothing here has been acquired or measured unless it says so. An entry leaves
this file only when it has both a measurement and a licence that fits the
project's openly-licensed-only stance.

Surveyed 2026-09-17 unless noted.

## Japanese, resting on Article 13

Article 13 of the Copyright Act is set out in `DATA.md`. These are the sources
that rest on it and are still outside the corpus. e-Gov laws also rest on
Article 13 and are in `DATA.md`.

| Source | Article 13 basis | Size | Blocker |
| --- | --- | --- | --- |
| Court judgments | 3 | 4.3 GB via NII | terms of use, see below |
| Notices and circulars, via the gazette | 2 | unknown | PDF only, see below |

## Japanese, resting on open licences

| Source | Licence | Size | Blocker |
| --- | --- | --- | --- |
| Government white papers, e-Stat | Government Standard Terms of Use 2.0, CC BY 4.0 compatible | unknown | not surveyed |
| J-STAGE open access | CC BY and variants | unknown | licences are per journal, see below |
| Common Corpus Japanese subset | mixed open | unknown | not investigated |

## Japanese text where place names occur in context

Added 2026-09-17. A gazetteer supplies the list of names; it does not supply
the sentences they live in. What the corpus needs is text where 渋谷区 appears
because something happened there, not because it is row 13,002 of a table.

Japan has an unusual advantage here: several of the densest such document sets
fall outside copyright entirely under Article 13.

| Source | Basis | Density | Blocker |
| --- | --- | --- | --- |
| Municipal ordinances and rules (例規集) | Article 13(1): 条例 are 法令 | very high | no single bulk endpoint |
| Ministry notices on road and district boundaries | Article 13(2) | very high | published via the gazette, PDF |
| JMA disaster-prevention XML (warnings and advisories) | Government Standard Terms of Use | very high | highly repetitive |
| Municipal newsletters, regional disaster plans | varies by municipality | high | terms differ per site, not bulk-usable |

### Municipal ordinances

Downgraded 2026-09-17. This was listed as the way to get Japanese place names
into context. Japanese Wikipedia turned out to have done it already: every one
of the 1,892 municipalities occurs, with a median of 448 occurrences across
1.39 million articles. See `TOKENIZER.md`.

Still interesting on its own terms, as Article 13 text in a register nothing
else in the corpus covers, but no longer load-bearing. Ordinances and rules are 法令 and therefore outside
copyright under Article 13(1), and every one of roughly 1,741 municipalities
publishes a full 例規集. The text is dense in local place names by
construction: ordinances define districts, enumerate town and section names,
and give the addresses of facilities.

Cross-municipality entry points exist, though none is confirmed to offer bulk
download:

- 条例Webアーカイブデータベース, Doshisha University: <https://jorei.slis.doshisha.ac.jp/>
  search, comparison between municipalities, and repealed or amended ordinances
- 全国条例データベース powered by eLen, Kagoshima University
- 全国自治体例規集, RILG: <https://www.rilg.or.jp/htdocs/main/zenkoku_reiki/zenkoku_link.html>
  a link directory to each municipality

Since the underlying text cannot be the subject of copyright, collecting it
from the municipalities' own sites carries no licensing obstacle. The obstacle
is engineering: 1,741 sites, several hosted 例規集 systems, no common format.

### JMA disaster-prevention XML

Warnings and advisories are issued per municipality, so each bulletin puts
municipality names into a sentence. A history database from December 2012
onwards is hosted by NII at <https://agora.ex.nii.ac.jp/cps/weather/report/>,
and the feed is registered on the e-Gov data portal.

The caveat is repetition. Tens of thousands of near-identical bulletins
deduplicate down to very little, and the context they supply, while real, is
one context. Worth taking for name coverage, not for variety.

## Detail on each blocker

Court judgments. Attractive in principle, since Article 13(3) puts them
outside copyright entirely, and they would be the Japanese counterpart to the
case law that bulks out the Common Pile. Two routes, neither usable as-is.

- NII's 日本の判例HTMLデータ (<https://www.nii.ac.jp/dsc/idr/rdata/HANREI/>) is
  67,313 cases from 1947 to 2026, 4.3 GB of HTML with CSV metadata and RDF.
  This is by far the largest Japanese source found. But use is restricted to
  academic research, access requires an application and review, the unit of
  provision is the laboratory, and annual research reports are required.
  Redistribution and model training are not addressed. Not usable for a
  publicly released model.
- courts.go.jp publishes judgments as PDFs with no bulk interface, so the
  direct route means building a scraper and a PDF pipeline. Open data for
  civil judgments is still in progress: of roughly 225,000 civil judgments in
  2022, commercial databases carried 10,000 to 20,000.

Official gazette (官報), which would cover the notices and circulars of
Article 13(2). Digitised from 2025-04-01 under the law on publication of the
gazette, with the old internet edition closed on 2025-03-31. Published as PDF
at <https://kanpou.npb.go.jp/> with no stated bulk or text interface.

J-STAGE. The WebAPI returns article listings rather than full text, and CC
licences are set per journal rather than per article and include NC and ND
variants that this project cannot use. Usable, but only after filtering.

Government white papers and e-Stat: not yet surveyed.

## What this means for the total

Resolved on 2026-09-17. The two available groups were downloaded and measured,
and have moved to `DATA.md`: e-Gov laws at 0.426 B tokens and the six Wikimedia
sister projects at 0.129 B together. The Japanese total is 2.685 B.

The prediction made here before measuring was 2.3 to 2.5 B. It was wrong, and
wrong because e-Gov was estimated at 60 M tokens from the compressed archive
size and turned out to be 426 M.

Of what is left in this file, the one source that would change the total
materially is the 4.3 GB of court judgments, and the convenient route to it is
closed by its terms of use.

## Candidates needing a legal determination before use

- Diet proceedings (国会会議録). Whether these fall under Article 13 paragraph 2
  is not obvious and has not been determined.
- National Diet Library digitised full text. Licensing varies by collection.

## Geospatial

### Rejected: the OSM database

The OSM Wiki and the OSM database are separate things under separate terms.
The wiki text is CC BY-SA 2.0. The database is ODbL, and the OSM Foundation
has a stated position on machine learning: a training set that is a substantial
extraction of OSM data is a Derivative Database and must be offered under ODbL
if used publicly, the model must be attributed in its documentation, and the
model's predictions are not implicated.

That is workable in itself, but it does not combine well with publishing a
single token store containing every source at once, which would make the whole
store a Derivative Database. The OSM database is therefore excluded from the
pretraining corpus. Nothing is lost for the stated goal, because tagging
conventions live in the wiki, not in the database.

### Deferred to post-training: the place hierarchy

Knowing that Harajuku is in Shibuya, Shibuya in Tokyo, and Tokyo in Japan is
bounded, enumerable and structured: 47 prefectures, roughly 1,700
municipalities, roughly 200,000 chome. It is available as CC0 from Wikidata
(P131) and as CC BY 4.0 from Geolonia's japanese-addresses.

Revised 2026-09-17. The original decision here was to keep it out of
pretraining entirely, on the grounds that two hundred thousand templated
sentences are a rounding error inside a corpus of tens of billions of tokens,
and that reliable recall needs varied exposure to each fact rather than
repetition of one template.

That reasoning holds against generating a synthetic corpus of sentences. It
does not hold against a compact grouped listing, which is a different artefact
with a different cost. A gazetteer emitted as `北海道: 札幌市 函館市 小樽市 …`
encodes containment as adjacency, runs to a few megabytes rather than a
generated corpus, and is exactly the form the OSM Wiki already carries on its
per-prefecture list pages. Adjacency inside a cleanly grouped list is the
signal a transformer takes up most readily, and it carries less ambiguity than
the same relation scattered through prose.

So the split is: a grouped gazetteer listing belongs in pretraining, and
structured instruction data after pretraining consolidates it. Not one or the
other.

The related observation that language models answer spatial questions by
recombining linguistic patterns rather than reasoning over geometry points the
same way, as does the fact that this project's own deployment target already
runs Nominatim, Overpass and a planet extract, which answer such questions
exactly.

Coordinates, geohashes and hierarchical cell indices are therefore out of scope
for v0.3.

### Undecided: existing datasets in this repository's orbit

Three datasets already published by this project's author cover adjacent
ground and are cached locally:

| Dataset | Rows | Shape |
| --- | --- | --- |
| `yuiseki/osm-tag-corpus` | 31,913 | tag, key, value, lang, title, description, lead_sentences, related_terms, on, implies, status, count_all. 9,467 English and 2,033 Japanese. |
| `yuiseki/text2geoql` | 4,815 | input, output. Natural language to query. |
| `yuiseki/osm-tokyo23-questions` | 131 | question templates |

`osm-tag-corpus` overlaps the wiki extraction above and is the tidied form of
the same material, so using both would duplicate. `text2geoql` is
post-training material, not pretraining material.

One provenance question is open: `count_all` looks like it comes from taginfo,
which derives from the OSM database. Aggregate counts are unlikely to be a
substantial extraction under ODbL, but this project restricts itself to openly
licensed sources, so the origin of that column has to be recorded here before
the dataset is used.
