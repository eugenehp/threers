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
threers-swipe eval    --model swipe.safetensors --words en=words_en.txt \
  --held-out test.jsonl --futo-layout qwerty.json --n 50000

# Score it; try a word.
cargo run -p threers-swipe --release -- eval --model swipe.safetensors --words en=words_en.txt
cargo run -p threers-swipe --release -- decode --model swipe.safetensors --words en=words_en.txt --word keyboard
```

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
threers-swipe = { version = "0.1.0", features = ["keyboard"] }
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
