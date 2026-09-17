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

## Open

- The selection threshold for `-geo`, and whether the result is large enough to
  stand alone. Being measured.
- Whether the token counts quoted are stated against a named tokenizer. They
  are tokenizer-dependent, and the figures so far use Qwen3 as an instrument.
- The selection threshold for `-geo`, and whether one threshold suits every
  language or each needs its own.
