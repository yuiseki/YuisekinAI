# Data information

This is the Data Information component required by the Open Source AI
Definition: a description of the data the system is trained on, its provenance,
and where it can be obtained. It records what is in the corpus and what has been
measured.

Sources that are surveyed, blocked, deferred or rejected live in
`DATA_candidate.md`. Nothing moves from that file into this one without a
measurement and a licence.

This project additionally restricts itself to public domain and openly licensed
text. The OSAID does not require this; it is a separate choice.

Status: no training run has been made. Every measurement below is dated.

## The constraint

English openly licensed text is abundant. Japanese openly licensed text is not.
This asymmetry, not compute and not disk, is what shapes the rest of the design.

Storage is not a constraint. A 50B-token store is 100 GB at two bytes per
token, against 908 GB free on the largest local volume, and the raw text is
never written to disk.

## The regime

English openly licensed text is abundant, Japanese is not. That asymmetry is a
studied regime rather than a problem to be papered over with a mixing ratio,
and the published findings are specific enough to design against.

The M-cubed scaling law (<https://arxiv.org/abs/2410.12325>) covers exactly this
shape: a scarce target language beside an abundant one. Its central result is
that mixing both languages through a single stage is never the optimal recipe.
The choice is between monolingual single-stage training, when the target corpus
is large, and multilingual two-stage training, when it is scarce. Which one
applies is set by the scarcity ratio, the size of the target corpus against the
compute-optimal corpus size for the budget.

For the two-stage recipe the paper puts almost no target language in the first
stage and concentrates it in the last, and finds the ratios in between make
little difference. The optimal number of epochs over the scarce corpus is
approximately the compute-optimal corpus size divided by the target corpus
size, so it is a consequence of scarcity rather than a free parameter. Related
work on mixture pretraining under data constraints
(<https://arxiv.org/abs/2605.12715>) reports scarce corpora being reused 15 to
20 times before the returns stop justifying it, which is far past the roughly
four epochs that earlier data-constrained work is usually remembered for.

The design consequences:

- English carries the warmup and the stable phase; Japanese is concentrated in
  the decay. This is a stage boundary, not a mixture weight.
- The Japanese corpus is repeated many times, and how many is derived from the
  scarcity ratio rather than chosen.
- The model size is bounded by the Japanese data, not by the budget.

What this costs in practice is worked out under "What 2.13 B tokens permits"
below, now that the Japanese corpus has been measured.

## English

### Common Pile v0.1 / Comma v0.1 training dataset

- Source: <https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset>
- Paper: <https://arxiv.org/abs/2506.05209>
- Basis: public domain and openly licensed text, assembled and documented by
  EleutherAI across roughly 30 sources.
- Size: 463.6B raw tokens in the main stage, 1,034.4B effective after the
  published mixture weights, plus 176.2B raw in the cooldown stage.
- Largest components: USPTO 157.4B, stackv2_edu 67.8B, peS2o 43.3B tokens.
- Note: effectively English-only. The component sources are US patents, case
  law, arXiv, PubMed and Stack Exchange.

Only a fraction of this is needed. A 0.5B model trained at 100 tokens per
parameter consumes 50B tokens, which is about 11% of the main stage.

## Japanese

Two distinct legal bases apply, and they are not interchangeable.

### Basis 1: Article 13 of the Japanese Copyright Act

Article 13 places four categories outside the reach of copyright entirely. They
are not licensed; they cannot be the subject of rights in the first place.

1. The Constitution and other laws and regulations
2. Notices, directives, circulars and similar issued by national or local
   government bodies, or by incorporated administrative agencies
3. Judgments, decisions, orders and rulings of the courts, and the
   determinations of administrative agencies made through procedures equivalent
   to judicial proceedings
4. Translations and compilations of the above, produced by those same bodies

This is the Japanese counterpart to the US government works that make up much
of the Common Pile, and it is the strongest foundation available here.

In the corpus, measured below:

| Source | Article 13 basis | Licence also |
| --- | --- | --- |
| e-Gov laws and regulations | 1 | Government Standard Terms of Use 2.0 |

Court judgments and the gazette rest on the same article but are not in the
corpus. What blocks each is in `DATA_candidate.md`.

Note that government white papers and academic works published by government
bodies fall outside Article 13. They are covered by basis 2 instead.

### Basis 2: open licences

In the corpus, measured below:

| Source | Licence |
| --- | --- |
| Wikipedia ja | CC BY-SA 4.0 |
| Aozora Bunko | public domain |
| Wikisource, Wiktionary, Wikibooks, Wikinews, Wikivoyage, Wikiquote ja | CC BY-SA 4.0, Wikisource originals often public domain |
| OSM Wiki | CC BY-SA 2.0 |

Further openly licensed Japanese sources that are surveyed but not yet in the
corpus are in `DATA_candidate.md`.

### Measured

Wikipedia ja and Aozora Bunko were sized from dataset metadata, which is exact,
and converted at a bytes-per-token ratio measured on a streamed sample. The
others were downloaded and processed in full. Tokens are counted with the Qwen3
tokenizer, used as a measuring instrument only.

| Source | Measured | Records | Text | Bytes/token | Japanese | Tokens |
| --- | --- | --- | --- | --- | --- | --- |
| Wikipedia ja, 20231101 | 2026-09-16 | 1,389,467 | 7.04 GB | 3.55 | 76% | 1.970 B |
| e-Gov laws, data of 2026-09-16 | 2026-09-17 | 10,679 laws | 1.69 GB | 3.97 | 87% | 0.426 B |
| Aozora Bunko, cleaned | 2026-09-16 | 16,951 | 0.71 GB | 3.91 | 89% | 0.160 B |
| Wikimedia sister projects, ja | 2026-09-17 | 561,666 pages | 718 MB wikitext | | | 0.129 B |
| OSM Wiki, JA namespace | 2026-09-16 | 4,132 | 15.5 MB | 3.83 | 41% | 0.0005 B |
| | | | | | | **2.685 B** |

The sister projects in detail:

| Dump | Pages | Wikitext | Prose kept | Japanese | Tokens |
| --- | --- | --- | --- | --- | --- |
| Wikisource | 22,805 | 335.9 MB | 81% | 82% | 72.27 M |
| Wiktionary | 506,781 | 214.6 MB | 33% | 36% | 23.46 M |
| Wikibooks | 23,380 | 133.5 MB | 78% | 59% | 28.57 M |
| Wikinews | 4,613 | 15.3 MB | 49% | 72% | 2.05 M |
| Wikivoyage | 2,461 | 15.4 MB | 42% | 63% | 1.76 M |
| Wikiquote | 1,626 | 3.8 MB | 85% | 63% | 0.89 M |

Two of those need a second look before use rather than after. Wiktionary is
only 36% Japanese by character and keeps only a third of its bytes as prose,
because it is a dictionary full of foreign headwords and heavy templates.
Wikibooks at 59% carries a lot of code and formulas. Neither is disqualified,
but neither is 23 M and 29 M tokens of Japanese prose either.

e-Gov deserves a note on size. The bulk download page reports 308 MB for all
law XML, which is the compressed figure; the archive is 323 MB and expands to
3.61 GB of XML across 10,679 laws, yielding 1.69 GB of text. An earlier
estimate in this project put it near 60 M tokens. It is 426 M.

Caveat on the two metadata-derived rows: the bytes-per-token sample is the
first 400 records of each stream, not a random draw.

Sources not yet in this table are in `DATA_candidate.md`.

### What 2.69 B tokens permits

This is the number that bounds the model size, so it is worth working through.

Assume a budget of 100 tokens per parameter, which is where small models are
trained now, and take the reported useful reuse of a scarce corpus at 15 to 20
epochs.

Spreading Japanese evenly across the whole run, which is the design the M-cubed
result told us not to use, the project stops at roughly 0.3 B parameters: a
0.5 B model would need 18.6 epochs over the Japanese corpus and a 1 B model
37.2.

Concentrating Japanese in stage two instead, where stage two is 15% of the
budget:

| Model | Budget | Stage two | Japanese epochs needed |
| --- | --- | --- | --- |
| 0.1 B | 10 B | 1.5 B | 0.6x, Japanese does not even fill it |
| 0.3 B | 30 B | 4.5 B | 1.7x |
| 0.5 B | 50 B | 7.5 B | 2.8x |
| 1.0 B | 100 B | 15 B | 5.6x |
| 2.0 B | 200 B | 30 B | 11.2x |
| 3.0 B | 300 B | 45 B | 16.8x |

Reading the ceiling the other way: at 15 epochs of reuse the Japanese corpus
supports a 2.7 B model, and at 20 epochs a 3.6 B one.

So the two-stage recipe is not a refinement that buys a little loss. It is the
difference between a Japanese corpus that bounds this project at 0.3 B
parameters and one that does not bind until an order of magnitude later. That
is worth restating because the design adopted it on the strength of a paper,
before there was any measurement to check it against.

Two things this does not say. Stage two need not be purely Japanese, and
making it so would cost English and the OSM Wiki conventions that stage one
taught. And the budget of 100 tokens per parameter is an assumption about
money, not a property of the data.

## Geospatial

v0.3 aims at a model that knows OpenStreetMap tagging conventions. That is
language knowledge: what `amenity=public_bath` means, when `shop` is used
instead of `amenity`, what the Japanese community's conventions are. It is
written as prose, in the OSM Wiki, and it belongs in pretraining.

### Measured, 2026-09-16

From the 2026-01-30 full-history dump (6.3 GB gzipped), taking the latest
revision of each page. The dump carries roughly 88 revisions per page, so
almost all of its size is history: 292,914 pages and 803 MB of current
wikitext come out of it.

Namespaces kept, after flattening wikitext to prose. Token counts are measured
with the Qwen3 tokenizer used only as a measuring instrument, not as a
candidate for this project.

| Namespace | Contents | Pages | Wikitext | Prose kept | Tokens |
| --- | --- | --- | --- | --- | --- |
| 0 | main, English, includes Key: and Tag: | 81,960 | 350.5 MB | 55% | 84.4 M |
| 3000 | Proposal, how conventions were decided | 2,581 | 15.3 MB | 33% | 1.25 M |
| 212 | JA | 4,132 | 15.5 MB | 29% | 1.28 M |
| 12 | Help | 33 | 11 KB | 46% | negligible |
| | | | | | **87 M** |

Namespaces not kept: User (127 MB), the wikibase Item and Property spaces
(57 MB), Talk (41 MB), the other language namespaces DE, RU, ES, FR, IT, NL
(104 MB together), Template, Category, File and Module.

### The Japanese OSM Wiki is close to empty

Of the 1.28 M tokens in the JA namespace, 41.5% contain Japanese characters.
The rest is untranslated English left in place by partial translations. The
genuinely Japanese OSM Wiki is therefore about **0.53 M tokens**.

That is not a corpus. Even repeated twenty times it is 10 M tokens. Japanese
tagging conventions cannot be taught from the Japanese OSM Wiki alone, and the
design has to account for that rather than assume the JA namespace carries its
weight. The English OSM Wiki, at 84 M tokens, is where the conventions actually
live.

### Cleaning is not a regex

The OSM Wiki keeps its tagging knowledge inside wikitables and inside
`{{Tag}}` and `{{Key}}` templates, not in paragraphs. A first pass with a
Wikipedia-style cleaner that strips tables and templates reported that 89% of
the corpus was markup. It was not: the cleaner was deleting the tag
descriptions, which are table cells.

`src/data/osm_wiki/clean.py` unwraps rather than strips, and
`tests/test_osm_wiki_clean.py` pins the behaviour so the same mistake cannot
return silently.

### What this does decide about pretraining

One thing, and it is irreversible after the fact. The tokenizer is fixed by
pretraining, and post-training will see the world through whatever vocabulary
it is given. If Japanese place names and OSM tag keys fragment badly, the
post-training stage starts at a disadvantage it cannot undo.

So the held-out text used to choose the vocabulary must include Japanese place
names and OSM tag keys, not only general Japanese and English prose. See the
open questions in `DESIGN.md`.

## Unresolved

### Share-alike

Wikipedia ja is CC BY-SA, and it is likely to be the single largest Japanese
source. Whether the share-alike obligation propagates to trained weights is
unsettled. The Common Pile includes share-alike text; this project has not yet
taken a position. A position must be taken and recorded here before any weights
are released, because it determines what the parameters can be released under,
and the OSAID requires those to be under OSI-approved terms.

### Volume

Measured at 2.685 B Japanese tokens.

An earlier revision of this file predicted the total would land between 2.3 and
2.5 B once the available candidates were added. It did not: e-Gov alone was
seven times the estimate made for it and carried the total past that range on
its own. The lesson is narrow and worth keeping, since it has now happened
twice in this document: estimate sizes from compressed archives at your peril,
and measure before writing a number down.

What remains open is not the figure but two things downstream of it: the budget
in tokens per parameter, which is a question about money rather than about
data, and where the stage boundary falls, which the figure constrains but does
not fix.
