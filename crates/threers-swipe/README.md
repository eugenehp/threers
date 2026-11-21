# threers-swipe

Gesture (swipe) typing for the threers on-screen keyboard
([`threers_ui::virtual_keyboard`](../threers-ui)): draw a word across the
keys without lifting the finger, get the word. Trains the model and decodes
with it. Pure Rust, no ML framework; without the default `parallel`
feature (rayon) it builds for wasm32.

## How it works

One small model serves every keyboard. It never learns where the keys are:
at each point of a swipe it says whether the finger meant a key there,
where it was aiming, and how sure it is. The keyboard's own key centres turn
that into a score per key. The seven built-in layouts, phone and tablet
keyboards, one-handed and floating ones all share one set of weights, and
a layout added later works without retraining (retraining makes it better).

| Part | Module | From |
|---|---|---|
| Key centres from `KeyboardConfig` + `KeyboardLayout` (`KeyboardConfig::key_boxes`), or from on-screen rects | `geometry` | threers-ui |
| 64 points evenly spaced along the path: position, direction, turn, speed, progress | `trace` | FUTO Swipe |
| Temporal conv net (dilated, residual) → gate, blank, aim offset, spread; key logit = gate − distance² / 2σ² | `model` | FUTO's layout-agnostic head, simplified to one aim point |
| CTC loss, gradients written out by hand and checked against finite differences | `ctc`, `model` | |
| Made-up swipes: aim scattered round each key, bends, minimum-jerk speed, Fitts' law timing | `synth` | IndicSwipe |
| Keys and swipe stretched, sheared, turned, mirrored together; swipes run backwards | `augment` | FUTO Swipe |
| Recorded swipes: this crate's JSON lines, or FUTO's `swipe.futo.org` (MIT) | `data` | |
| Beam search over a trie of the language's words, weighted by frequency | `decode` | CTC prefix beam, lexicon-constrained |
| AdamW, warmup + cosine, clipping, batches over all cores | `train` | FUTO's recipe |

## Weights

[`weights/swipe-en-small.safetensors`](weights/) (897 KB): English, trained
on FUTO's 933 k recorded swipes plus made-up ones on the threers
keyboards. On FUTO's held-out test split (49 661 swipes) it reads
**88.1%** of words first and 93.2% in the top three with a lexicon of the
training sentences' words, 87.8% / 92.9% with one from unrelated
sentences, a word the lexicon lacks counting as a miss. How it was made
and what those numbers do and do not say: [weights/README.md](weights/README.md).

```rust,ignore
let model = threers_swipe::Model::load("crates/threers-swipe/weights/swipe-en-small.safetensors")?;
```

## Training

You need a word list per language: one word a line, optionally with a
count (`the 23135851`), most common first. Any frequency list works (for
example the `hermitdave/FrequencyWords` lists, CC-BY-SA, cut to the top
50 000).

```bash
# Made-up swipes only: English and German, phone + tablet keyboards.
cargo run -p threers-swipe --release -- train \
  --words en=words_en.txt --words de=words_de.txt \
  --steps 20000 --out swipe.safetensors

# FUTO's corpus (MIT, 933 k English swipes, 5.2 GB): their layout file,
# the word list from the training sentences, dev held out.
B=https://huggingface.co/datasets/futo-org/swipe.futo.org/resolve/main
curl -LO $B/train.jsonl -LO $B/dev.jsonl -LO $B/test.jsonl
curl -L $B/swipe-5/layouts/qwerty.json -o qwerty.json
threers-swipe check   --recorded dev.jsonl --futo-layout qwerty.json   # ~0.3 / 0.5 keys
threers-swipe lexicon --from train.jsonl --out words_en.txt            # 69 k words
threers-swipe train   --words en=words_en.txt --recorded train.jsonl \
  --held-out dev.jsonl --futo-layout qwerty.json \
  --steps 50000 --batch 128 --lr 3e-3 --recorded-share 0.85 --out swipe.safetensors
# stopped? the same command carries on from swipe.state.safetensors
threers-swipe tune    --model swipe.safetensors --words en=words_en.txt \
  --held-out dev.jsonl --futo-layout qwerty.json            # pick --lm-weight
threers-swipe eval    --model swipe.safetensors --words en=words_en.txt \
  --held-out test.jsonl --futo-layout qwerty.json --n 50000 --lm-weight 0.75

# Score it; try a word.
cargo run -p threers-swipe --release -- eval --model swipe.safetensors --words en=words_en.txt
cargo run -p threers-swipe --release -- decode --model swipe.safetensors --words en=words_en.txt --word keyboard
```

### How training works

Each step (`train::Trainer::train_step`):

1. **Draws a batch** (`--batch`, 128 for FUTO). Each swipe is a recorded
   one with chance `--recorded-share` (0.85 for FUTO), else a made-up one:
   a random word of the `--words` list on one of the language's phone,
   tablet and large-tablet keyboards (`synth`: aim scattered round each
   key, bends between keys, minimum-jerk speed, Fitts' law timing).
   Made-up swipes keep the model on the threers keyboards' shapes, not
   only FUTO's canvas.
2. **Moves swipe and keyboard together** (`augment`): stretched 0.85–1.15
   across and 0.75–1.3 down, sheared, turned up to 8°, mirrored
   left-right (half the time) and top-bottom (a fifth), the swipe a
   little off the keys; one in ten runs backwards with its word
   reversed; one in ten loses its times and three in ten get them rounded
   to 60 Hz frames, as a UI stamps events. The model learns shapes
   against the keys it is shown, not one layout.
3. **Targets the key path**: one class per key the word crosses, a key
   typed twice running once (a swipe cannot show "ll"), apostrophes
   passed over.
4. **CTC loss** over the 64 points, gradients written out by hand
   (checked against finite differences), summed over the batch on every
   core in fixed groups of 8, so the sums are the same on any number of
   threads. Swipes too short for their word are left out.
5. **AdamW**: gradients averaged and clipped to norm 1, weight decay
   1e-4, the rate warmed up over the first 5% of steps to `--lr` (3e-3 for
   FUTO) and down a cosine to 2e-5.
6. **Moving average of the weights** (`--ema`, 0.999, ramped in over the
   first steps). The average is what is evaluated and exported: smoother
   than the last step's weights.

Every `--eval-every` steps (and at the end) `train` writes:

| File | What |
|---|---|
| `--state` (default `<out>.state.safetensors`) | weights, AdamW's two moments, the moving average and the step: the run carries on from here exactly, bit for bit, when `train` is run again with the same flags (`--fresh` starts over) |
| `<out>.last.safetensors` | the averaged model now |
| `--out` | the averaged model that scored best so far on the held-out sample |

With `--held-out` the score is top-1 over a fixed `--eval-n` swipes (1000)
at `--lm-weight` (0.75), the same swipes every time, so checkpoints
compare. Each swipe draws its randomness from the seed, the step and its
place in the batch, so a run is reproducible.

Scores count a word the lexicon lacks as a miss (no lexicon decoder can
type it), and say how many there were. Pick `--lm-weight` on dev with
`tune`, never on test. With a lexicon from FUTO's training sentences,
the test's sentences come from the same pool, so its frequencies fit the
test unusually well; a lexicon from other sentences
(`lexicon --from other.txt`, plain text works) is the fairer check.

Speed and memory: the FUTO run is ~0.1 s a step of 128 on 6 threads of a
loaded M-series Mac (50 000 steps ≈ 1.5 h), 933 k swipes in about 1 GB.

A self-contained try, no files needed (200 words, a couple of minutes):

```bash
cargo run -p threers-swipe --release --example swipe_train_synth
```

Weights are safetensors with the `ModelConfig` in the metadata
(`input.weight`, `blocks.N.conv.weight`, `blocks.N.mix.weight`,
`head.weight`, …), so another runtime (RLX included) can load them.

## On the keyboard (opt-in)

Gesture typing is off unless asked for: the `keyboard` feature of this
crate, which turns on the `swipe` feature of `threers-ui`. Without it the
keyboard has no swipe code at all and a slide types the key it ends on.

```toml
threers-swipe = { version = "0.1.1", features = ["keyboard"] }
```

```rust,ignore
use threers_swipe::keyboard::SwipeTyping;
use threers_swipe::{Lexicon, Model};

SwipeTyping::new(Model::load("swipe.safetensors")?)
    .language("en", Lexicon::load("words_en.txt")?)
    .language("de", Lexicon::load("words_de.txt")?)
    .attach(&virtual_keyboard);
```

A press on a letter becomes a swipe once the finger has gone 0.9 of a key
width (before a hold offers accents); taps and short slides type as they
did. On lifting, the best word is typed (a space before it after a word,
a capital when Shift is on) and the next three are offered in the bar,
where a tap swaps them in. The keys come with each swipe as drawn, so a
split or floating keyboard decodes as it shows. Any other decoder can be
given to `VirtualKeyboard::set_swipe_decoder` directly.

### Next words

Two next-word models read the sentence instead of the finger, each behind
its own feature: an interpolated Kneser–Ney n-gram (`ngram`) and a small
feed-forward neural model trained and run by RLX (`rlx`: three words'
embeddings, one hidden layer, a softmax over the vocabulary). Either one,
or both through `Interpolated`, does two jobs: swipes are ranked by
`log p(word | the words before)` instead of the word's frequency ("to"
after "I'm going", "too" after "me"), and the bar offers the next word
after a space and completions while a word is typed
(`VirtualKeyboard::set_predictions`).

```rust,ignore
use threers_swipe::{ngram::NgramModel, neural::NeuralLm, Interpolated};

SwipeTyping::new(Model::load("swipe.safetensors")?)
    .language("en", Lexicon::load("words_en.txt")?)
    .next_words("en", Interpolated::new()
        .with(NgramModel::load("en.ngram")?, 0.5)
        .with(NeuralLm::load("en.next.safetensors")?, 0.5))
    .attach(&virtual_keyboard);
```

```bash
D=/path/to/futo; B="cargo run -p threers-swipe --release --features ngram,rlx --bin threers-next-word --"
$B ngram  --text $D/sentences.txt --order 3 --out en.ngram            # ~1.5 s for 99 k sentences
$B neural --text $D/sentences.txt --vocab 20000 --steps 40000 --f16 1 --out en.next.safetensors
$B eval   --text $D/test.jsonl --ngram en.ngram --neural en.next.safetensors
$B swipes --model swipe.safetensors --words words_en.txt --held-out $D/test.jsonl \
  --futo-layout $D/qwerty.json --ngram en.ngram --neural en.next.safetensors
```

`eval` scores the bar on held-out sentences (the next word after a space,
and after its first letter); `swipes` decodes FUTO's recorded swipes after
the words before them in their sentence (each line's `sentence` and
`word_idx`), against the word list alone, over a range of `lm_weight`s.
`demo` prints the bar as sentences are typed, and the held-out swipes the
words before them changed.

In the browser: `web/swipe-demo/` is the keyboard with all of it — swipe,
tap, and the bar's next words from the n-gram and the neural model mixed
(RLX runs in the page's wasm). Its models are in `web/swipe-demo/models/`,
with how each was made and how they score.

```bash
SERVE=1 web/swipe-demo/build.sh   # then open http://localhost:8787/swipe-demo/
```

A third kind learns: `UserModel` is one person's own 3-gram, taught each
word as they finish it (`SwipeTyping::user(lang, model, 0.2)`), so their
names and phrases come up as a phone's keyboard's do. It knows only what
they typed, takes nothing from the language's models for other words, and
saves as text (the demo keeps it in `localStorage`). On text in a style
the language's model had not seen, learning at 0.2 took the next word's
place in the bar's three from 46.5% to 52.7% of the time after its first
letter (`eval --learn 0.2`).

`threers_swipe::prelude::*` is one import for decoding and next words;
`threers_ui::prelude::*` has the text field and touch input beside it.

To decode outside the keyboard (no feature needed):

```rust,ignore
use threers_swipe::{Decoder, KeyboardGeometry, Lexicon, Model, TouchPoint};

let keyboard = KeyboardGeometry::from_config(&config, &layout, panel_width);
let decoder = Decoder::new(&keyboard, &Lexicon::load("words_en.txt")?);
let swipe: Vec<TouchPoint> = touches
    .iter()
    .map(|t| { let (x, y) = keyboard.to_units(t.x, t.y); TouchPoint::new(x, y, t.time) })
    .collect();
let words = decoder.decode(&Model::load("swipe.safetensors")?, &swipe, 3);
```

## Where it stands

- The model, loss, decoder and both data readers are tested (`cargo test
  -p threers-swipe`), gradients against finite differences, CTC against
  brute force.
- On the keyboard it is tested with a decoder that answers fixed words
  (`threers-ui --features swipe`) and with a tiny model trained in the
  test (`--features keyboard`); not yet with a full model by a person.
- FUTO's lines are read a line at a time (933 k swipes in about 1 GB and
  20 s); the row spacing comes from each line's canvas, the keys from
  their `layouts/qwerty.json`. Words lose their sentence punctuation
  ("lava." → "lava"); apostrophes need no key ("don't" is d-o-n-t).
- Recorded swipes exist only for English (FUTO). The other six layouts
  learn from made-up swipes until real ones are logged
  (`Recording::to_json`).
