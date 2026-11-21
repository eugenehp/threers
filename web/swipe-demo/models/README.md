# Models for the swipe and next-word demo

What `web/swipe-demo/` loads, kept in the tree so the demo runs from a
checkout (`web/swipe-demo/build.sh`). All English.

| File | What | Size | Made from |
|---|---|---|---|
| `swipe.safetensors` | the swipe model (`threers_swipe::Model`, small: 224 k weights, 50 k steps) | 0.9 MB | FUTO's recorded swipes |
| `words_en.txt` | the words a swipe can be, with counts | 0.7 MB | FUTO's training sentences |
| `en.ngram` | next words: Kneser–Ney 3-gram, 40 k words, binary form | 5.5 MB | FUTO's prompt sentences |
| `en-chat.ngram` | next words: Kneser–Ney 3-gram, 40 k words, trigrams seen twice or more | 5.7 MB | Tatoeba's everyday sentences |
| `en.next.safetensors` | next words: the neural model (`NeuralLm`, small: 3 words × 32 → 128 → 20 k words, 3.3 M weights), f16 | 6.7 MB | FUTO's prompt sentences |

The page mixes the three next-word models 0.4 : 0.4 : 0.2 (FUTO n-gram,
Tatoeba n-gram, neural), the shares chosen on held-out sentences of both
kinds: FUTO's are encyclopedic, Tatoeba's are what people say, and a
message is mostly the second. What the person types is learnt into a
fourth model, 0.2 of the mix (`threers_swipe::user`), kept by the page in
`localStorage`; `?forget=1` clears it.

## Sources

- FUTO's `swipe.futo.org` corpus, MIT:
  <https://huggingface.co/datasets/futo-org/swipe.futo.org>. The swipe
  model trains on `train.jsonl`; the next-word models on its 99,269 prompt
  sentences less the 731 its dev and test swipes were typed in
  (`swipe5_en_minus_held.txt`), so no dev or test sentence is in them.
- Tatoeba's English sentences, CC BY 2.0 FR, © the Tatoeba contributors:
  <https://tatoeba.org>, from
  <https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2>
  (2,041,577 sentences, 2026-11-20). Sentence ids ending 00 are held out for
  test, ending 01 for dev; the 2,000,760 others, less any sentence also in
  dev or test, are `en-chat.ngram`'s.

```bash
B="cargo run -p threers-swipe --release --features ngram,rlx --bin threers-next-word --"
awk -F'\t' '$1 % 100 > 1 { print $3 }' eng_sentences.tsv > tatoeba-train.txt
$B ngram  --text $FUTO/swipe5_en_minus_held.txt --order 3 --vocab 40000 --out en.ngram
$B ngram  --text tatoeba-train.txt --order 3 --vocab 40000 --min-count 2 --out en-chat.ngram
$B neural --text $FUTO/swipe5_en_minus_held.txt --vocab 20000 --steps 40000 --f16 1 --out en.next.safetensors
```

## How they score (test splits)

The bar, over 20,000 words of each test set (an unknown word counts as a
miss): how often the next word is among its three, after a space and
after the word's first letter.

| | FUTO test | | Tatoeba test | |
|---|---|---|---|---|
| | after a space | first letter | after a space | first letter |
| FUTO 3-gram alone | 27.8% | 49.9% | 20.7% | 45.5% |
| the page's mix | 27.4% | 50.0% | **32.8%** | **65.3%** |

Swipes: 10,000 of FUTO's recorded test swipes decoded after the words
before them in their sentence, `lm_weight` 0.75 (96.7% of their words are
in the word list, the most top-1 can reach):

| Ranked by | Top-1 | Top-3 |
|---|---|---|
| the word list's frequency alone | 88.1% | 93.2% |
| FUTO 3-gram | 89.4% | 93.6% |
| Tatoeba 3-gram | 87.5% | 93.1% |
| neural | 89.3% | 93.6% |
| the page's mix | **89.7%** | **93.7%** |

The f16 neural weights score as the f32 ones did, to the digit; pruning
the Tatoeba trigrams seen once (20.7 MB → 5.7 MB) cost about a point after
the first letter on its dev set.
