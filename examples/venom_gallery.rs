//! The fx style on anything: four shapes wearing [`Venom`] in four looks, two
//! wearing [`Gloss`], all on one [`Stage`] under one pointer.
//!
//! ```text
//! cargo run --release --example venom_gallery
//! cargo run --release --example venom_gallery -- --autoplay
//! cargo run --release --example venom_gallery -- --png out/venom_gallery.png
//! ```
//!
//! Each shape is a [`Figure`] — [`Shape::pill`], [`Shape::circle`],
//! [`Shape::star`], [`Shape::from_svg_path`], [`Shape::rect`] — and the
//! captions are a component written here, in the example, to show that a new
//! [`Component`] (and an [`Effect`] that draws nothing) takes a dozen lines.
//!
//! With `--png` there is no window: each venom gets its own scripted hand
//! ([`Stage::update_each`]), timed so they are caught in different moments —
//! holding, reaching, grabbing from the side and snapping back.

use threers::fx::{
    Anchor, Backdrop, Component, Effect, EffectInput, Figure, Gloss, GlossStyle, Layout, Look, Shape, Stage,
    StageInput, Venom, VenomStyle,
};
use threers::math::Vector2;
use threers::overlay::Overlay;
use threers::prelude::*;
use threers::sketch::{sketch, Frame, SketchOptions};

include!("venom_common.inc");

const VIEW: (f32, f32) = (1100.0, 640.0);
const HEART: &str = "M0 -3C-4 -8 -10 -3 0 6C10 -3 4 -8 0 -3Z";

/// A line of grey text under a point — a component with no surface.
struct Caption {
    text: &'static str,
    anchor: Anchor,
    at: Vector2,
}

impl Component for Caption {
    fn layout(&mut self, layout: &mut Layout) {
        self.at = self.anchor.resolve(layout.viewport);
    }
    fn shape(&self) -> Shape {
        Shape::circle(self.at, 0.0)
    }
    fn paint(&self, o: &mut Overlay, dpr: f32) {
        o.no_stroke();
        o.text_weight(400);
        o.text_size(13.0 * dpr);
        o.fill(Color::from_hex(0x6b6b70));
        o.text_centered(self.at.x * dpr, self.at.y * dpr, self.text);
    }
    fn interactive(&self) -> bool {
        false
    }
}

/// An effect that wears nothing.
struct Bare;

impl Effect for Bare {
    fn update(&mut self, _input: &EffectInput) {}
}

/// The grid cell `(i, j)` of three by two, as a fraction of the viewport.
fn cell(i: usize, j: usize) -> Vector2 {
    Vector2::new(0.18 + 0.32 * i as f32, 0.28 + 0.46 * j as f32)
}

/// The gallery: the venom elements first (their ids, for the scripted hands),
/// then the gloss ones and the captions.
fn gallery(scene: &mut Scene) -> (Stage, Vec<threers::fx::ElementId>) {
    let mut stage = Stage::new();
    stage.set_backdrop(scene, Backdrop::STUDIO);
    let venom = |look: Look| Venom::new(VenomStyle::default().with_look(look));
    let place = |shape: Shape, i: usize, j: usize| Figure::new(shape).at(Anchor::Fraction(cell(i, j)));
    let heart = Shape::from_svg_path(HEART, Vector2::ZERO, 9.0, 4.0).expect("heart");
    let venoms = vec![
        stage.add(scene, place(Shape::pill(Vector2::ZERO, 220.0, 64.0), 0, 0), venom(Look::LACQUER)),
        stage.add(scene, place(Shape::circle(Vector2::ZERO, 58.0), 1, 0), venom(Look::CHROME)),
        stage.add(scene, place(Shape::star(Vector2::ZERO, 5, 78.0, 34.0, 7.0), 0, 1), venom(Look::GOLD)),
        stage.add(scene, place(heart, 1, 1), Venom::new(VenomStyle::default().with_look(Look::TOXIC).with_strands(5))),
    ];
    let gloss = |look: Look| Gloss::new(GlossStyle::default().with_look(look));
    stage.add(scene, place(Shape::rect(0.0, 0.0, 180.0, 110.0, 22.0), 2, 0), gloss(Look::LACQUER));
    stage.add(scene, place(Shape::circle(Vector2::ZERO, 50.0), 2, 1), gloss(Look::GOLD));
    let captions = [
        ("Figure(pill) · Venom · LACQUER", 0, 0, 0.12),
        ("Figure(circle) · Venom · CHROME", 1, 0, 0.16),
        ("Figure(star) · Venom · GOLD", 0, 1, 0.2),
        ("Figure(svg) · Venom · TOXIC", 1, 1, 0.16),
        ("Figure(rect) · Gloss · LACQUER", 2, 0, 0.16),
        ("Figure(circle) · Gloss · GOLD", 2, 1, 0.14),
    ];
    for (text, i, j, below) in captions {
        let c = cell(i, j);
        let anchor = Anchor::Fraction(Vector2::new(c.x, c.y + below));
        stage.add(scene, Caption { text, anchor, at: Vector2::ZERO }, Bare);
    }
    (stage, venoms)
}

fn main() {
    if let Some(path) = arg("--png") {
        headless(&path);
        return;
    }
    let autoplay = flag("--autoplay");
    sketch(SketchOptions::new("threers — fx on anything").size(VIEW.0 as u32, VIEW.1 as u32), move |s| {
        let (mut stage, _) = gallery(s.scene);
        let mut last_seen = f32::NEG_INFINITY;
        move |f: &mut Frame| {
            if f.pointer_inside {
                last_seen = f.elapsed;
            }
            if !f.pointer_inside && autoplay && f.elapsed - last_seen > 2.5 {
                // a ghost hand tours the gallery while the pointer is away
                let dpr = f.scale_factor;
                let view = Vector2::new(f.width as f32 / dpr, f.height as f32 / dpr);
                let pointer = Some(ghost_at(f.elapsed * 0.7, view.x, view.y));
                f.cursor_visible = true;
                stage.update(f.scene, f.overlay, StageInput { viewport: view, dpr, dt: f.delta, pointer, pointer_down: false });
            } else {
                stage.sketch_frame(f);
            }
        }
    });
}

fn headless(path: &str) {
    use threers::renderer::headless::HeadlessRenderer;

    let (view, dpr) = (Vector2::new(VIEW.0, VIEW.1), 2.0f32);
    let (pw, ph) = ((view.x * dpr) as u32, (view.y * dpr) as u32);
    let mut hr = HeadlessRenderer::builder().size(pw, ph).build().expect("headless renderer");
    let mut scene = Scene::new();
    let mut camera = PerspectiveCamera::new(60.0, view.x / view.y, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 2.0);
    camera.look_at(Vector3::ZERO);
    let (mut stage, venoms) = gallery(&mut scene);
    let mut overlay = Overlay::new(pw, ph);
    let input = StageInput { viewport: view, dpr, dt: 1.0 / 60.0, pointer: None, pointer_down: false };
    stage.update(&mut scene, &mut overlay, input);

    // A hand per venom, from above and to the right of it unless noted.
    let scripts: Vec<Vec<(f32, Vector2)>> = venoms
        .iter()
        .enumerate()
        .map(|(i, &id)| {
            let (lo, hi) = stage.shape(id).expect("laid out").bounds();
            let top_right = Vector2::new(hi.x - (hi.x - lo.x) * 0.2, lo.y);
            let at = |dx: f32, dy: f32| top_right + Vector2::new(dx, dy);
            match i {
                // drifts in and holds
                0 => vec![(0.0, at(200.0, -200.0)), (1.6, at(80.0, -120.0)), (2.6, at(30.0, -70.0)), (9.0, at(25.0, -60.0))],
                // hovers in range without coming close enough to grab
                1 => vec![(0.0, at(260.0, -200.0)), (2.0, at(150.0, -110.0)), (9.0, at(140.0, -105.0))],
                // comes in from the side, low, and holds
                2 => {
                    let side = Vector2::new(hi.x + 30.0, (lo.y + hi.y) * 0.5);
                    vec![(0.0, side + Vector2::new(240.0, 60.0)), (2.0, side + Vector2::new(70.0, 0.0)), (9.0, side + Vector2::new(50.0, -10.0))]
                }
                // grabbed, then yanked away just before the shot
                _ => vec![(0.0, at(200.0, -200.0)), (1.5, at(50.0, -90.0)), (3.0, at(40.0, -80.0)), (3.4, at(150.0, 230.0)), (9.0, at(150.0, 230.0))],
            }
        })
        .collect();

    let mut t = 0.0f32;
    while t < 3.55 {
        overlay.clear();
        stage.update_each(&mut scene, &mut overlay, input, &|id| {
            venoms.iter().position(|&v| v == id).map(|i| scripted(&scripts[i], t))
        });
        t += 1.0 / 60.0;
    }
    let mut rgba = hr.render_to_rgba(&mut scene, &camera);
    composite(&mut rgba, overlay.pixels());
    write_png(path, pw, ph, &rgba);
    for &id in &venoms {
        println!("  {:?}", stage.effect::<Venom>(id).map(|v| v.state()));
    }
}
