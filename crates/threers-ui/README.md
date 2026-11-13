# threers-ui

The retained UI of [threers](https://github.com/eugenehp/threers): boxes
styled with a subset of CSS, laid out with flexbox, painted through any 2D
`Painter`, hit-tested for input, and described to assistive technology as an
accessibility tree with WCAG contrast checks.

It also has:

- **Event dispatch** with capture and bubble phases, focus, keyboard
  activation and Tab order, wheel scrolling and pointer capture.
- **A keymap** turning keystrokes, chords included, into actions of your own
  types, scoped by key contexts.
- **Recorded scenes**, for a renderer to draw: threers draws them on the GPU
  in one instanced call.
- **Layout caching** for subtrees that have not changed.
- **Generational node ids**, so a removed node's id never names its
  replacement.
- **Virtual lists, deferred and anchored popups** and an indexed hit test.
- **Rich text runs, images, box shadows, wavy underlines** and GPU
  surfaces for video.
- **Touch gestures, IME composition, drag and drop, tooltips, tab order**
  and cursors.
- **Springs, a cross-frame text cache, element state, container queries**,
  an inspector, a frame-stats overlay and random-tree testing.
- **Hover, active and focus styles with transitions**, property animations,
  and drawing only when something changed.
- **CSS Grid, scrollbars, text selection with the pointer**, modals that
  trap focus, click-outside, and keymaps from JSON with context
  expressions.
- **Native screen readers through AccessKit** (the `accesskit` feature).

`web/ui-demo` in the repository runs all of this in a browser.

It knows nothing of the renderer. `threers::ui` re-exports it and adds
`OverlayPainter`, which paints onto threers' CPU overlay; in a browser,
`canvas::CanvasPainter` paints onto the page's own 2D canvas and
`a11y_dom::DomAccessibility` puts the tree in the page for screen readers.

```rust
use threers_ui::{ui, FixedWidthText, Ui};

let mut tree = Ui::new();
let root = tree.root();
let bar = ui!(tree, root,
    div(display: flex; height: 20px; background: #222) {
        div(padding: 0 2px; font-size: 13px) { "x" }
        div(flex: 1) {}
    }
);
tree.layout(200.0, 100.0, &mut FixedWidthText(7.0));
assert_eq!(tree.rect(bar).height, 20.0);
```
