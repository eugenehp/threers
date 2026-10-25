# Interactive effects: `threers::fx`

Glossy liquids that react to the pointer, on buttons, figures and anything
else with a shape. Built on [signed distance fields](sdf.md).

<a href="images/venom_gallery.jpg"><img src="images/venom_gallery.jpg" width="640"></a>

## Three ideas

| | | ships with |
|---|---|---|
| **Component** | what an element *is*: its `Shape`, its content, how it responds | `Button`, `Figure` |
| **Effect** | what it *wears*: a behaviour that draws a `Surface` and paints detail | `Venom`, `Gloss` |
| **Stage** | the elements together: layout, hit-testing and events, the scene quads, paint order, one `Cursor` | |

Any component wears any effect. A new component works with every effect, and
a new effect with every component.

```rust
use threers::fx::{Anchor, Backdrop, Button, EventKind, Figure, Gloss, GlossStyle, Look, Shape, Stage, Venom, VenomStyle};
use threers::math::Vector2;
use threers::sketch::{sketch, Frame, SketchOptions};

sketch(SketchOptions::new("fx"), |s| {
    let mut stage = Stage::new();
    stage.set_backdrop(s.scene, Backdrop::STUDIO);
    let go = stage.add(s.scene, Button::new("let's go"), Venom::default());
    stage.add(
        s.scene,
        Figure::new(Shape::star(Vector2::ZERO, 5, 70.0, 30.0, 6.0)).at(Anchor::Fraction(Vector2::new(0.5, 0.25))),
        Venom::new(VenomStyle::default().with_look(Look::GOLD)),
    );
    stage.add(s.scene, Figure::new(Shape::circle(Vector2::new(120.0, 120.0), 40.0)), Gloss::new(GlossStyle::default().with_look(Look::CHROME)));
    move |f: &mut Frame| {
        for event in stage.sketch_frame(f) {
            if event.element == go && event.kind == EventKind::Click {
                println!("clicked");
            }
        }
    }
});
```

Positions are CSS pixels — logical, top-left origin, y down. Outside a
`sketch`, call `stage.update(scene, overlay, StageInput { viewport, dpr, dt,
pointer, pointer_down })` each frame. It returns the events, puts each
effect's surface on its quad in the scene, and paints the overlay.

## Venom

<a href="images/venom_button.jpg"><img src="images/venom_button.jpg" width="480"></a>

A liquid that hunts for the pointer. Bring it near and the surface swells
toward it while pseudopods feel about in its direction. Get close and they
lunge, each grabbing a grip of the cursor (a finger of the cartoon glove),
with a pulse running along them while a coat of the surface creeps over the
hand. Pull away and they stretch, snap and slump back; the body wobbles and
throws up spikes and droplets. `stage.effect::<Venom>(id).state()` reports
`Idle`, `Seeking`, `Grabbing`, `Holding` or `Snapping`.

| `VenomStyle` | default | |
|---|---|---|
| `look` | `Look::LACQUER` | colour and reflections |
| `strands` | 7 | thick strands, 2 to 7 |
| `hairs`, `webs` | true | fine hairs between strands; threads strung across while holding |
| `sense` | 290 | px from the outline at which it starts reaching |
| `reach` | 140 | px at which it lunges and grabs |
| `release` | 215 | px at which it snaps; kept at least `reach` |
| `strand_width` | 1.0 | thickness multiplier |
| `damping` | 0.92 | velocity kept per frame: lower is thicker, more syrupy |
| `gravity` | 260 | px/s² on the strands |
| `mound` | 1.0 | how far the surface swells toward the pointer |
| `spikes`, `droplets` | true | what a snap throws up |
| `bevel` | 32 | deepest the round profile goes: big shapes get a pillowed edge and a flat top |
| `coat` | true | the surface creeping over the cursor while it holds |

Change it while it runs: `stage.effect_mut::<Venom>(id).unwrap().style.strands = 3`.

## Gloss

The same material, still: the shape in a `Look`, its reflections leaning
with the pointer, and a gentle bulge toward it when it comes within
`GlossStyle::reach`. Use it for cards, badges and icons that should match a
venom button without hunting.

## Looks

A `Look` is a base colour under a Fresnel reflection of a small studio: a
softbox above, a window to the left, a fill to the right, the floor below,
and the room behind the viewer. It adds two glints and a contact shadow.

| | base | reflects | F0 |
|---|---|---|---|
| `Look::LACQUER` | near black | cool white | 0.05 |
| `Look::CHROME` | dark grey | white | 0.55 |
| `Look::GOLD` | dark amber | warm gold | 0.6 |
| `Look::TOXIC` | green | pale green | 0.04 |

```rust
let ink = Look::LACQUER.with_base_hex(0x101848).with_tint_hex(0xc8d4ff);
let style = VenomStyle { look: ink, strands: 4, damping: 0.88, ..VenomStyle::default() };
```

`Look::named("chrome")` reads a preset from a string.

## Shapes

| `Shape` | |
|---|---|
| `Shape::pill(centre, w, h)` | buttons |
| `Shape::rect(x, y, w, h, radius)` | cards, panels |
| `Shape::circle(centre, r)` | avatars, icons |
| `Shape::star(centre, points, outer, inner, radius)` | badges |
| `Shape::polygon(points, radius)` | any outline, corners rounded by `radius`; up to 128 points |
| `Shape::from_svg_path(d, origin, scale, radius)` | the first subpath of SVG path data |

A `Figure` with an `Anchor` re-centres its shape every layout, so it follows
the window. `Outline::new(&shape)` walks any shape's edge by arc length, which
is how venom roots its strands whatever the shape.

## Cursors

`Cursor::default()` is the cartoon glove. The stage draws it once, over every
element, and hides the system cursor while it does. `Cursor::Hotspot` keeps
the system cursor and draws nothing. Every cursor has eight *grips* that
effects can reach for (`cursor.grip(hotspot, i)`), and effects can leave a coat
on it through `Effect::decorate_cursor`.

## Adding an effect

```rust
use threers::fx::{Effect, EffectInput, Look, Surface};

/// A shape that throbs.
struct Pulse { surface: Option<Surface>, t: f32 }

impl Effect for Pulse {
    fn update(&mut self, input: &EffectInput) {
        self.t += input.dt;
        let k = 1.0 + 0.04 * (self.t * 6.0).sin();
        self.surface = Some(Surface::new(input.shape.scaled(k), Look::TOXIC));
    }
    fn surface(&self) -> Option<Surface> {
        self.surface.clone()
    }
}
```

`update` gets the element's shape, the pointer, the cursor, and whether the
element is hovered or pressed. `surface` returns what to draw: the shape plus
anything grown out of it (strands as chains of `Blob`s, `Spike`s, droplets, a
`Mound`), shaded in a `Look`. `paint` and `paint_over_cursor` draw detail on
the overlay, under and over the cursor. The surface renderer is shared, so a
new effect draws on every backend the built-in ones do.

## Adding a component

```rust
use threers::fx::{Component, ComponentInput, Layout, Shape};
use threers::math::Vector2;

/// A round toggle that grows while hovered.
struct Toggle { at: Vector2, grow: f32 }

impl Component for Toggle {
    fn layout(&mut self, layout: &mut Layout) {
        self.at = layout.viewport * 0.5;
    }
    fn shape(&self) -> Shape {
        Shape::circle(self.at, 30.0 * (1.0 + 0.1 * self.grow))
    }
    fn update(&mut self, input: &ComponentInput) {
        let target = if input.hovered { 1.0 } else { 0.0 };
        self.grow += (target - self.grow) * (input.dt * 10.0).min(1.0);
    }
}
```

The stage hit-tests the shape, raises `Enter`, `Leave`, `Press`, `Release`
and `Click`, and passes `hovered` and `pressed` to both the component and its
effect. `paint` draws content (a label, an icon) over the surface and under
the effect's detail. Return `false` from `interactive` for something
decorative.

## Backends

Surfaces are `ShaderMaterial`s whose WGSL is built on `sdf::WGSL`, so they run
on wgpu everywhere: Metal, Vulkan, DX12, WebGPU, and WebGL2 (the arrays reach
it through the preamble's texture fallback). threers' own Metal renderer
(`--features metal`) runs no WGSL. It recognizes surfaces and draws them with
its MSL port (`src/metal/fx_surface.metal`), blending on encoded values as the
wgpu path does. `tests/fx_metal.rs` renders one stage through both and holds
them to within 1.5 levels on average; they agree to 0.06. The overlay detail
is a CPU canvas and goes wherever the overlay does.

## Examples

```text
cargo run --release --example venom_button                     # a Button wearing Venom
cargo run --release --example venom_button -- --look gold --strands 4
cargo run --release --example venom_gallery                    # six shapes, both effects
cargo run --release --example venom_gallery -- --autoplay      # a ghost hand tours them
cargo run --release --example venom_button -- --png out/venom  # headless frames
```
