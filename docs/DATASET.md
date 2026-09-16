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
| `YuisekinText-tiny` | the openly licensed Japanese pretraining corpus |
| `YuisekinText-geo` | the geographically dense subset, separable and usable alone |
| `YuisekinAI-*` | models trained on them |

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

To be adoptable on its own it needs three things beyond the text:

- Provenance and licence per document, so that a user can trace what they are
  mixing into their own corpus.
- A reproducible selection criterion, shipped with the gazetteer it was applied
  with. Without the 63,587-name candidate list, there is no way to say why a
  document is in.
- A frozen, citable artefact. A dataset that grows quietly cannot be cited.

## Open

- The selection threshold for `-geo`, and whether the result is large enough to
  stand alone. Being measured.
- Whether the token counts quoted are stated against a named tokenizer. They
  are tokenizer-dependent, and the figures so far use Qwen3 as an instrument.
- Whether the English side is part of `YuisekinText-tiny` or stays a pointer to
  the Common Pile.
