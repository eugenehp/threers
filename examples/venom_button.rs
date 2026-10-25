//! A glossy black button that hunts for a host: a [`Button`] wearing
//! [`Venom`] on a [`Stage`].
//!
//! ```text
//! cargo run --release --example venom_button
//! cargo run --release --example venom_button -- --look chrome --strands 5
//! cargo run --release --example venom_button -- --autoplay      # a ghost hand plays with it
//! cargo run --release --example venom_button -- --cursor        # keep the system pointer
//! cargo run --release --example venom_button -- --png out/venom # headless frames, no window
//! cargo run --release --example venom_button -- --png out/venom --dpr 1 # at one device pixel per CSS pixel
//! ```
//!
//! Bring the pointer near and the surface swells toward it while pseudopods
//! ooze out and feel about in its direction. Get close and they lunge, each
//! coiling round a finger of the cartoon glove, a pulse running along them
//! while the goo creeps over the hand. Pull away and they stretch, snap and
//! slump back; the body wobbles and throws up spikes. Press the button and it
//! gives; release over it and it clicks.
//!
//! The whole example is the stage's setup: everything else is
//! [`threers::fx`]. `--look` takes `lacquer`, `chrome`, `gold` or `toxic`; see
//! `examples/venom_gallery.rs` for the style on other shapes and other effects.

use threers::fx::{Backdrop, Button, Cursor, EventKind, Look, Stage, StageInput, Venom, VenomStyle};
use threers::math::Vector2;
use threers::overlay::Overlay;
use threers::prelude::*;
use threers::sketch::{sketch, Frame, SketchOptions};

include!("venom_common.inc");

const LABEL: &str = "let\u{2019}s collaborate";

fn style() -> VenomStyle {
    let mut style = VenomStyle::default();
    if let Some(look) = arg("--look").and_then(|n| Look::named(&n)) {
        style.look = look;
    }
    if let Some(n) = arg("--strands").and_then(|s| s.parse().ok()) {
        style.strands = n;
    }
    style
}

/// The stage: the studio backdrop and one button.
fn stage(scene: &mut Scene) -> Stage {
    let cursor = if flag("--cursor") { Cursor::Hotspot } else { Cursor::default() };
    let mut stage = Stage::new().with_cursor(cursor);
    stage.set_backdrop(scene, Backdrop::STUDIO);
    stage.add(scene, Button::new(LABEL), Venom::new(style()));
    stage
}

fn main() {
    if let Some(prefix) = arg("--png") {
        headless(&prefix);
        return;
    }
    let autoplay = flag("--autoplay");
    let frame_limit: Option<u64> = arg("--frames").and_then(|s| s.parse().ok());

    sketch(SketchOptions::new("threers — venom button").size(900, 560), move |s| {
        let mut stage = stage(s.scene);
        let mut last_seen = f32::NEG_INFINITY;

        move |f: &mut Frame| {
            if f.pointer_inside {
                last_seen = f.elapsed;
            }
            let events = if !f.pointer_inside && autoplay && f.elapsed - last_seen > 2.5 {
                // a ghost hand plays until the pointer comes back
                let dpr = f.scale_factor;
                let view = Vector2::new(f.width as f32 / dpr, f.height as f32 / dpr);
                let input = StageInput {
                    viewport: view,
                    dpr,
                    dt: f.delta,
                    pointer: Some(ghost_at(f.elapsed, view.x, view.y)),
                    pointer_down: false,
                };
                f.cursor_visible = true;
                stage.update(f.scene, f.overlay, input)
            } else {
                stage.sketch_frame(f)
            };
            for event in events {
                if event.kind == EventKind::Click {
                    println!("clicked: {LABEL}");
                }
            }
            if frame_limit.is_some_and(|n| f.frame + 1 >= n) {
                f.exit = true;
            }
        }
    });
}

/// No window: a scripted hand drifts in, is grabbed, holds over the button and
/// is yanked away; a frame is written at each moment worth seeing.
fn headless(prefix: &str) {
    use threers::renderer::headless::HeadlessRenderer;

    let dpr = arg("--dpr").and_then(|s| s.parse().ok()).unwrap_or(2.0f32);
    let view = Vector2::new(900.0, 560.0);
    let (pw, ph) = ((view.x * dpr) as u32, (view.y * dpr) as u32);
    let mut hr = HeadlessRenderer::builder().size(pw, ph).build().expect("headless renderer");
    let mut scene = Scene::new();
    let mut camera = PerspectiveCamera::new(60.0, view.x / view.y, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 2.0);
    camera.look_at(Vector3::ZERO);
    let mut stage = stage(&mut scene);
    let mut overlay = Overlay::new(pw, ph);

    // one frame without a pointer to lay the button out, then the script
    let input = |pointer| StageInput { viewport: view, dpr, dt: 1.0 / 60.0, pointer, pointer_down: false };
    stage.update(&mut scene, &mut overlay, input(None));
    let button = stage.ids().next().expect("the button");
    let (lo, hi) = stage.shape(button).expect("its shape").bounds();
    let corner = Vector2::new(lo.x + (hi.x - lo.x) * 0.75, lo.y - 10.0);
    let at = |dx: f32, dy: f32| corner + Vector2::new(dx, dy);
    let keys = [
        (0.0, at(260.0, -260.0)),
        (1.6, at(120.0, -150.0)),
        (2.6, at(40.0, -75.0)),
        (4.4, at(30.0, -60.0)),
        (5.0, at(300.0, -240.0)),
        (8.0, at(300.0, -240.0)),
    ];
    let shots = [(1.4, "reach"), (2.4, "seek"), (3.6, "hold"), (4.9, "stretch"), (5.3, "snap")];
    let mut t = 0.0f32;
    for (when, name) in shots {
        while t < when {
            overlay.clear();
            stage.update(&mut scene, &mut overlay, input(Some(scripted(&keys, t))));
            t += 1.0 / 60.0;
        }
        let mut rgba = hr.render_to_rgba(&mut scene, &camera);
        composite(&mut rgba, overlay.pixels());
        write_png(&format!("{prefix}_{name}.png"), pw, ph, &rgba);
        println!("  t={t:.2}s {:?}", stage.effect::<Venom>(button).map(|v| v.state()));
    }
}
