# YuisekinAI v0.3 design

Status: draft. Nothing in this document has been implemented yet.

This document records the design decisions for the v0.3 rewrite. The v0.2
pipeline is preserved on the `legacy/2024-pipeline` branch and is not carried
forward.

## Goals

1. Satisfy the [Open Source AI Definition](https://opensource.org/ai/open-source-ai-definition):
   publish the data information, the complete training and inference code, and
   the model parameters, all under OSI-approved terms.
2. Train only on openly licensed or public domain text. This is stricter than
   the OSAID requires, and is a deliberate choice rather than an obligation.
3. Cover Japanese as well as English. See `DATA.md` for why this is the
   binding constraint on everything else.

## Why v0.2 is not being extended

`src/run_clm.py` is HuggingFace's fine-tuning example. Everything downstream
inherits that assumption: the model is loaded from a checkpoint directory
rather than constructed, the whole corpus is tokenized through `datasets.map`
before training starts, and the run is configured by a large JSON of
`TrainingArguments`.

Pretraining from scratch needs different things:

- a fixed token stream with a deterministic order
- checkpoints that resume at token granularity, not epoch granularity
- a measurement of model FLOPs utilisation, so that rented GPU time can be
  budgeted before it is spent
- an optimiser that is not AdamW, which the `Trainer` optimiser plumbing
  does not accommodate cleanly

## Decided

### Token store

Corpora are streamed, tokenized on the fly, and appended to `uint16` memmap
shards. The intermediate plain-text files of v0.2 are not produced at all;
they were the reason tokenizer preparation ran out of 96 GB of RAM.

Each shard is accompanied by an index of document start offsets, so that
packing can avoid letting one document predict the next.

The element width is a tradeoff with the vocabulary size rather than a fixed
rule. A vocabulary below 65536 fits `uint16`, which puts a 50B-token store at
100 GB; above that the store is `uint32` and the same run costs 200 GB. There
is 908 GB free on the largest local volume, so either is affordable, and the
vocabulary should be chosen on how well it encodes Japanese and English rather
than to fit a width. For reference, PLaMo-13B uses 64K and LLM-jp-4 uses
between 100K and 256K.

The store lives outside the repository and outside any git working tree.

### Tokenizer

Trained in-project, not adopted. Existing open-weight tokenizers either do not
document their training data, which would import an undescribable component
into an OSAID system, or are tuned for English only.

SentencePiece, as in v0.2, with BPE and byte fallback so that there is no
unknown token. SentencePiece was not the cause of the v0.2 tokenizer problems.
The conversion to `XLNetTokenizer` was: that class contributes its own special
tokens and its own id conventions, which is where the mismatch between the
trained piece ids and the model's `bos_token_id` and `eos_token_id` came from.

So the conversion is removed from the training path entirely. The corpus-to-
memmap pass calls SentencePiece directly. A HuggingFace-compatible wrapper is
produced only at export time, for distribution, and is covered by a test that
checks it against the SentencePiece model it was built from.

Special token ids are asserted by a test, not assumed.

The model is also exported to the FlatBuffers format (`.spm.fb`) read by the
[SentencePiece Lite runtime](https://google.github.io/sentencepiece/lite/),
which is part of upstream SentencePiece and Apache-2.0 licensed. That runtime
is a ~45 KB stripped static library under 2k lines with no third-party
dependencies, not even Protobuf or Abseil, and it mmaps the model so that heap
use is 1.2 KB regardless of vocabulary size. It reports higher encode
throughput than both the original library and HuggingFace Tokenizers.

This matters twice. It makes the tokenizer deployable on the microcontroller
and single-board hardware this project's author already works with, which
changes what the released artefact is good for. And because the whole corpus is
encoded exactly once on the way into the token store, encode throughput is
directly the cost of the preprocessing pass;
`PretokenizeAtSafeBoundaries()` exists to parallelise it.

### Model

Defined in-project as an `nn.Module` rather than instantiated from
`MistralForCausalLM`. Three reasons: unused machinery such as sliding-window
attention can be dropped; the optimiser needs parameters classified into matrix
and non-matrix groups; and the forward pass should be legible in one file.

Decoder-only transformer, pre-norm RMSNorm, SwiGLU, rotary position embeddings,
grouped-query attention, QK normalisation, no biases. Embeddings are tied at
small sizes. No sliding window.

Weights are exported to HuggingFace format for distribution only. The training
format is the project's own.

### Precision

bf16 throughout. The v0.2 configuration used fp16 with loss scaling, and its
`adam_beta2: 0.9` and `adam_epsilon: 1.0e-4` are best read as divergence
workarounds. Those revert to conventional values under bf16.

### Optimiser

Muon for the two-dimensional hidden weights, AdamW for embeddings, the output
head, normalisation parameters and scalars.

With a calibrated expectation. "Fantastic Pretraining Optimizers and Where to
Find Them" (<https://arxiv.org/abs/2509.02046>) measures matrix-based
optimisers against an AdamW baseline that is tuned as carefully as they are,
and finds 1.4x at 0.1B parameters falling to 1.1x at 1.2B. The widely repeated
2x is an artefact of comparing against an undertuned AdamW. Muon is still worth
taking here, because this project sits at the small end where the gain is
largest, but it is worth roughly one and a half times the tokens, not twice.

Two methodological rules follow from the same paper and apply to every
comparison this project makes:

- Muon's hyperparameters are tuned for Muon. Inheriting AdamW's is the mistake
  that produces inflated numbers, in both directions.
- Optimisers are compared at the end of a completed decay, never at an
  intermediate checkpoint. Rankings flip during learning rate decay.

Dion3 (<https://arxiv.org/abs/2608.11612>) reports Muon's loss at up to 6x
lower optimiser step time. The saving is in distributed communication, so it is
not a priority at single-node scale, but it is the thing to watch if this
project ever runs wide.

### Schedule

Warmup-stable-decay rather than cosine, so that the number of steps need not be
fixed in advance, training can be extended, and checkpoints taken during the
stable phase remain usable.

The decay shape is 1-sqrt, which is the strongest of the published decay curves
rather than the linear or cosine decay that WSD is usually drawn with.

The stable phase is 80 to 90 percent of steps and is not cut short.

### Training stages

Not one mixture, but two stages. This is a data decision as much as a schedule
decision, and the reasoning is in `DATA.md`: for a scarce target language
alongside an abundant one, a single mixed stage is never the optimal recipe.

Stage one is English-dominant and occupies the warmup and the stable phase.
Stage two concentrates Japanese and occupies the decay.

The three things line up, which is the reason to build it this way: WSD's decay
phase, the second stage of the M-cubed recipe, and the separate cooldown corpus
that the Common Pile itself publishes alongside its main stage are the same
window.

### Checkpoints

A checkpoint stores the model, the optimiser state, the step, the number of
tokens consumed, the dataloader position and the RNG state, so that a run
resumes exactly where it stopped.

The OSAID asks for checkpoints from key intermediate stages of training and for
the final optimiser state. These are therefore release artefacts, not scratch
files, and the retention policy is chosen with that in mind.

### Configuration

One TOML file per run, read into dataclasses. Model shape, data mixture and
optimisation live together in a single readable file that doubles as the record
of the experiment. Model size is a set of numbers in that file, so that the same
code runs a 10M-parameter smoke test, a 0.1B run on two consumer GPUs, and a
larger run on rented hardware.

### Distribution

DDP first. FSDP2 when a run no longer fits, and only then.

This is also what torchtitan does: with `data_parallel_shard_degree = 1` it
falls back to DDP through `torch.distributed._composable.replicate`, and shards
with `fully_shard` above that. FSDP2 is now the centre of PyTorch-native
training, so the upgrade path is the well-trodden one.

### Tests

The pipeline is developed and tested on hardware that cannot pretrain. That is
workable because most of what can be wrong is testable at small scale:

- tokenizer round-trip, and the identity of every special token id
- memmap document offsets agreeing with the source document boundaries
- packing not leaking the tail of one document into the next
- a small model driven to near-zero loss on a single repeated batch

The last of these is the cheapest insurance against discovering, an hour into
rented GPU time, that the loss is not going down.

## Layout

```
configs/          one TOML per run
src/data/         corpus acquisition, one module per source
src/tokenize/     tokenizer training, and the corpus to memmap pass
src/model/        model definition
src/train.py      training loop
src/eval/         held-out loss and benchmarks
src/export/       HuggingFace format export for distribution
tests/
docs/DATA.md      the OSAID data information
docs/DESIGN.md    this file
```

## Open

- Where the boundary between stage one and stage two falls, how much Japanese
  stage one carries, and how many epochs the Japanese corpus is repeated for.
  All three follow from the scarcity ratio, which is not known until the
  Japanese corpus is measured. See `DATA.md`.
- Vocabulary size, and how it is split between the two languages. To be chosen
  by measuring compression on held-out text of both, not by assumption. The
  choice also sets whether the token store is `uint16` or `uint32`.
  The held-out text includes Japanese place names and OSM tag keys, because the
  vocabulary cannot be changed afterwards and post-training has to live with
  it. See the geospatial section of `DATA.md`.
- Unigram or BPE. BPE with byte fallback is the current assumption, but
  PLaMo-13B reaches 64K on Japanese with Unigram, so this deserves a
  measurement rather than a default.
- Whether share-alike licensed text is included. See `DATA.md`.
- Target model size and token budget. Deliberately deferred; the configuration
  format is designed so that this can be decided late.
