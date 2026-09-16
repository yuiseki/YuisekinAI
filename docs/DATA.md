# Data information

Status: draft inventory. Sizes marked "est." have not been measured. Nothing
here has been acquired or processed yet.

The Open Source AI Definition requires a complete description of the data used
to train the system, the provenance of that data, and a listing of where it can
be obtained. This document is that description, and it is expected to grow into
the authoritative record of the corpus rather than a summary written afterwards.

This project additionally restricts itself to public domain and openly licensed
text. The OSAID does not require this; it is a separate choice.

## The constraint

English openly licensed text is abundant. Japanese openly licensed text is not.
This asymmetry, not compute and not disk, is what shapes the rest of the design.

Storage is not a constraint. A 50B-token store is 100 GB at two bytes per
token, against 908 GB free on the largest local volume, and the raw text is
never written to disk.

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

| Source | Article 13 basis | Est. size | Status |
| --- | --- | --- | --- |
| e-Gov laws and regulations | 1 | est. 0.5 GB | not started |
| Court judgments (courts.go.jp) | 3 | unknown | not started |
| Notices and circulars | 2 | unknown | not started |

Note that government white papers and academic works published by government
bodies fall outside Article 13. They are covered by basis 2 instead.

### Basis 2: open licences

| Source | Licence | Est. size | Status |
| --- | --- | --- | --- |
| Wikipedia ja | CC BY-SA 4.0 | est. 6.4 GB | script exists (v0.2) |
| Aozora Bunko | public domain | est. 0.65 GB | script exists (v0.2) |
| Wikisource ja | CC BY-SA 4.0 and public domain | unknown | not started |
| Wiktionary / Wikibooks / Wikinews / Wikivoyage ja | CC BY-SA 4.0 | unknown | not started |
| Government white papers, e-Stat | Government Standard Terms of Use 2.0, compatible with CC BY 4.0 | unknown | not started |
| J-STAGE open access articles under CC | CC BY and variants | unknown | not started |
| Common Corpus Japanese subset | mixed open | unknown | not investigated |

The two v0.2 sizes are the byte counts of the extracted text files recorded in
`098_dataset_prepare.sh` on the `legacy/2024-pipeline` branch. They have not
been re-measured.

### Candidates needing a legal determination before use

- Diet proceedings (国会会議録). Whether these fall under Article 13 paragraph 2
  is not obvious and has not been determined.
- National Diet Library digitised full text. Licensing varies by collection.

## Geospatial

v0.3 aims at a model that knows OpenStreetMap tagging conventions. That is
language knowledge: what `amenity=public_bath` means, when `shop` is used
instead of `amenity`, what the Japanese community's conventions are. It is
written as prose, in the OSM Wiki, and it belongs in pretraining.

| Source | Licence | Est. size | Status |
| --- | --- | --- | --- |
| OSM Wiki, including JA: pages | CC BY-SA 2.0 | unknown | not started |

### Not the OSM database

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

### Place hierarchy is a post-training concern

Knowing that Harajuku is in Shibuya, Shibuya in Tokyo, and Tokyo in Japan is
bounded, enumerable and structured: 47 prefectures, roughly 1,700
municipalities, roughly 200,000 chome. It is available as CC0 from Wikidata
(P131) and as CC BY 4.0 from Geolonia's japanese-addresses.

It is not being put into the pretraining corpus. Two hundred thousand templated
sentences are a rounding error inside a corpus of tens of billions of tokens,
and reliable recall is reported to need varied exposure to each fact rather
than repetition of one template, so teaching it this way would mean generating
that variety first. Structured instruction data after pretraining is the
better-matched tool, and there is published work on exactly this shape of
problem.

The related observation that language models answer spatial questions by
recombining linguistic patterns rather than reasoning over geometry points the
same way, as does the fact that this project's own deployment target already
runs Nominatim, Overpass and a planet extract, which answer such questions
exactly.

Coordinates, geohashes and hierarchical cell indices are therefore out of scope
for v0.3.

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

The Japanese sources above are unlikely to exceed the low tens of billions of
tokens in total, and may be considerably less. The English side alone offers
463B.

This is not a problem to be worked around with a mixing ratio. It is a studied
regime, and the published findings are specific enough to design against.

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

So the design consequences are:

- English carries the warmup and the stable phase; Japanese is concentrated in
  the decay. This is a stage boundary, not a mixture weight.
- The Japanese corpus is repeated many times, and how many is derived from the
  scarcity ratio rather than chosen.
- The model size is bounded by the Japanese data, not by the budget.

Every one of these needs the size of the Japanese corpus as its input.
Measuring it is therefore the first task, and nothing downstream can be settled
until it is done.
