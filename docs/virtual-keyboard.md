# On-screen keyboard

Part of the [threers](../README.md) documentation.

`threers::ui::virtual_keyboard` draws a keyboard inside the UI: for a kiosk,
a car, a TV, a headset, or anywhere the platform's keyboard will not come
up. It looks and behaves like Apple's: letters type when the finger lifts,
the pressed letter shows above the finger, holding a letter offers its
accents, Backspace repeats and then deletes whole words, holding the space
bar turns the keys into a trackpad, Shift tapped twice locks, and sentences
start with a capital.

```bash
cargo run --features ui --example virtual_keyboards       # every preset and placement to PNG
cargo run --features ui --example keyboard_designer -- examples/keyboards/kiosk.kbd --watch
cargo run -p threers-ui --release --example keyboard_bench # what each press costs a frame
cargo run -p threers-ui --features sound --example keyboard_clicks
```

## The five-line version

```rust
use threers::ui::virtual_keyboard::{KeyboardType, VirtualKeyboard};

let kb = VirtualKeyboard::new(&mut ui, &mut input); // phone, tablet or large tablet, by window size
kb.set_enabled(true);                               // shows while a text field has focus
kb.set_type_for(email_field, KeyboardType::Email);  // @ and . by the space bar, no capitals
kb.set_type_for(pin_field, KeyboardType::NumberPad);
kb.set_context(|f: &TextField, _| Some(f.before_caret())); // capitals, ". ", suggestions
```

Each key types into whatever has focus as a hardware key would: the same
`Event::Key` events, so a field that handles the keyboard handles this one,
and focus stays where it is.

## Choosing a keyboard

| You want | Use |
|---|---|
| The right one for the window, again on resize | `VirtualKeyboard::new` (`Device::Auto`) |
| One device always | `VirtualKeyboard::with_device(.., Device::Tablet)` |
| Your own, everywhere | `VirtualKeyboard::with_config(.., config)` |
| A kind of field (email, URL, PIN, price, phone, search) | `kb.set_type_for(field, KeyboardType::…)` |
| A field with its own design | `kb.set_config_for(field, Some(config))` |
| One change to every keyboard shown | `kb.customize(\|c\| c.feedback.sound = false)` |

The presets are `KeyboardConfig::iphone`, `ipad`, `ipad_pro` (number row,
Tab, Caps Lock), `ipad_floating`, `number_pad`, `decimal_pad`, `phone_pad`,
`email` and `url`. Each is plain data: change any field.

## Where it sits

```rust
use threers::ui::virtual_keyboard::{Align, Placement};

Placement::docked()                      // across the bottom; the page above shrinks
Placement::one_handed(Align::Left)       // keys to one side, a › strip to go back
Placement::centered(640.0)               // docked at a width (anchored(Align::Right, 500.0) too)
Placement::floating(320.0)               // over the page; drag it by its handle
Placement::docked().split(200.0)         // halves at the sides, q w e r t | y u i o p
```

`kb.set_placement(Some(p))` moves it at run time, and so does a
`Key::place(p, label, name)` key.

## Designing one in text

A row is a line: each word is a key that types itself, `a|b` types `a` (or
`b` under Shift) and shows both, `{name}` is a special key, and `@` gives a
width in units (one unit is a letter key) or `*` for whatever the row leaves.

```text
{row:0}                                   # the language's first row of letters
{row:1}
{shift@1.3} {gap@0.2} {row:2} {gap@0.2} {backspace@1.3}
{123@1.25} {globe@1.25} {space} {return@2.5}
```

Special keys: `{shift}` `{caps}` `{backspace}` `{return}` `{space}` `{tab}`
`{123}` `{abc}` `{#+=}` `{globe}` `{dismiss}` `{gap}` `{row:N}`
`{key:ArrowLeft:←}` (a named key with a label) and `{action:emoji:☺}`, which
calls `kb.on_action`.

`KeyboardConfig::from_rows(&[...])` builds the letters page from rows like
these. `KeyboardConfig::parse(text)` reads a whole file, settings and all
(see [`examples/keyboards/kiosk.kbd`](../examples/keyboards/kiosk.kbd)), and
its errors give the line number. `keyboard_designer FILE --watch` checks the
file, prints each page as text and draws it to PNG every time you save:

```text
── letters
[ 1 ][ 2 ][ 3 ][ 4 ][ 5 ][ 6 ][ 7 ][ 8 ][ 9 ][ 0 ]
[ q ][ w ][ e ][ r ][ t ][ y ][ u ][ i ][ o ][ p ]
   [ a ][ s ][ d ][ f ][ g ][ h ][ j ][ k ][ l ]
[  ⇧  ] [ z ][ x ][ c ][ v ][ b ][ n ][ m ] [  ⌫  ]
[123 ][        space         ][ .com ][    go     ]
```

`config.validate()` lists what is wrong, with the page, row and key: a page
no key leads back from, `{row:5}`, a key 0 units wide. A debug build prints
these when the keyboard first shows the config. `kb.sketch()` returns the
page showing as text, for a test or a log.

## Feedback: how a press looks, sounds and feels

```rust
use threers::ui::virtual_keyboard::{KeyFeedback, PressStyle};

config.feedback = KeyFeedback {
    preview: true,           // the letter shown above the finger (phone)
    press: PressStyle::Swap, // a pressed key takes the other keys' colour (Dim, None)
    sound: true,
    volume: 0.6,
    haptics: true,
};
```

Each press is reported to `kb.on_feedback(|press| …)` with the click to
play (`KeySound::Letter`, `Delete` or `Modifier`, as Apple's three) and the
tap to give (`Haptic::Light`; `Medium` when Shift locks or a key's accents
come up). With the `sound` feature, `kb.play_clicks()` plays the clicks on
the default audio device. Without it, `KeySound::samples(rate)` gives the
click as mono samples for your own mixer.

## Text that knows the field

`kb.set_context(|state, field| Some(text_before_caret))` tells the keyboard
what is before the caret. With it:

- Shift comes on at the start of a sentence (`config.auto_capitalize`:
  `None`, `Sentences`, `Words`, `All`) and goes off again after one letter;
- two spaces after a word type “. ” (`config.period_shortcut`);
- `kb.set_suggestions(|word| …)` offers up to three words in the bar above
  the keys, and tapping one replaces the word being typed.

`TextField::before_caret()` and `TextArea::before_caret()` return that text.

## Speed

A key press is drawn without a layout pass and without changing the
tree's shape: the preview is the key's own bubble made opaque, every page
is built once and switched by display, and a change of Shift relabels only
the keys that change. Painters that redraw only what changed (`damage`)
redraw about the pressed key. Hidden boxes, and boxes clipped out of
sight, no longer widen a repaint.

`keyboard_bench` reports each operation's CPU time per frame (thread time,
so other work on the machine does not count), how often it ran a layout
pass, and how much of the window it damaged. Set `SHOW=1` to print what
changed and what widened the damage.
