# Swipe model weights

## `swipe-en-small.safetensors`

The `small` model (`ModelConfig::small()`: 96 channels, dilations
1, 2, 4, 8, 16, 1; 223 589 weights, f32, 897 KB), trained on English swipes.
Load with `Model::load` and decode with a `Decoder` over an English word
list.

SHA-256 `c4da00594a2249001d073dcc77b7833db66be6407142843e517a8e105158082e`.

### Data

- [`futo-org/swipe.futo.org`](https://huggingface.co/datasets/futo-org/swipe.futo.org)
  (MIT), its `swipe-1` training split: 933 502 English swipes on FUTO's
  QWERTY (`swipe-5/layouts/qwerty.json`), row spacing from each line's
  canvas. 85% of each batch.
- Made-up swipes (`synth`) of the words of FUTO's training sentences on
  the threers English keyboards (phone, tablet, large tablet): 15%.

### Training

```bash
threers-swipe train --words en=words_en_train.txt --recorded train.jsonl \
  --held-out dev.jsonl --futo-layout qwerty.json \
  --steps 50000 --batch 128 --lr 3e-3 --recorded-share 0.85
```

50 000 steps of 128 (6.4 M swipes, about 7 passes over the recordings),
AdamW, warmup + cosine, augmentation as in the crate README, on 6 CPU
threads of a shared Mac, about 1.5 h. These weights come from before the
moving average and exact resume were added: they are the last step's
weights, and the run was resumed once at step 35 000 with the optimiser's
moments restarted.

### Scores

FUTO's held-out `test.jsonl` (49 661 swipes after dropping words with
digits; different people from training), `--lm-weight 0.75` (chosen on
`dev.jsonl`, not on test). A word the lexicon lacks counts as a miss.

| Lexicon | top-1 | top-3 | words not in it |
|---|---|---|---|
| words of FUTO's training sentences (69 k) | 88.1% | 93.2% | 3.4% |
| words of other Common Voice sentences, none of dev's or test's (72 k) | 87.8% | 92.9% | 3.7% |

Made-up swipes on the threers iPhone keyboard: 96–97% top-1.

The test's sentences come from the same Wikipedia pool as training's, and
both lexicons are encyclopedic: typing chat or names will score lower.
FUTO report 84.4% top-1 for their own encoder on this split, with their
decoder; the decoders differ, so the two numbers are not a like-for-like
comparison.

### License

The weights are released under the repository's MIT license; the
training data is MIT (FUTO).
