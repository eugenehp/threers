//! Elements that flicker at a given frequency, in 3D and in the UI.
//!
//! On the left, scene objects with [`Object3D::flicker`] set, written into
//! the scene each frame by `Scene::apply_flicker`: a cube on and off at
//! 1 Hz, a sphere at 4 Hz on for a quarter of each cycle, a torus fading
//! as a 0.5 Hz sine (its material's opacity), and a point light pulsing at
//! 2 Hz. On the right, UI tiles given a flicker with `Ui::set_flicker`: the
//! same waveforms, a sine between 30% and 100%, and two 10 Hz tiles half a
//! cycle apart that alternate.
//!
//! ```bash
//! cargo run --release --features sketch,ui --example flicker
//! # Eight frames a sixteenth of a second apart, side by side, to a PNG:
//! cargo run --release --features sketch,ui --example flicker -- --strip
//! ```

use threers::math::flicker::Flicker;
use threers::overlay::Overlay;
use threers::prelude::*;
use threers::sketch::{sketch, Frame, SketchOptions};
use threers::style;
use threers::ui::{OverlayPainter, Ui};

const W: u32 = 1100;
const H: u32 = 560;

/// The scene objects and the waveforms they flicker with.
fn build_scene(scene: &mut Scene, camera: &mut PerspectiveCamera) {
    scene.background = Color::from_hex(0x0e1117);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.25));
    let mut sun = Object3D::light(DirectionalLight::new(Color::WHITE, 1.6));
    sun.position = Vector3::new(-3.0, 6.0, 5.0);
    scene.add(sun);

    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(20.0, 20.0),
        StandardMaterial::new(Color::from_hex(0x2a303c)).into(),
    ));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    scene.add(floor);

    let shapes: [(BufferGeometry, u32, f32, Flicker); 3] = [
        (
            BoxGeometry::new(1.0, 1.0, 1.0),
            0xe0874a,
            -2.4,
            Flicker::square(1.0),
        ),
        (
            SphereGeometry::new(0.6, 48, 24),
            0x4aa3e0,
            -0.8,
            Flicker::square(4.0).duty(0.25),
        ),
        (
            TorusGeometry::new(0.5, 0.2, 24, 64, std::f32::consts::TAU),
            0x7fd36b,
            0.8,
            Flicker::sine(0.5),
        ),
    ];
    for (geometry, color, x, flicker) in shapes {
        let mut o = Object3D::mesh(Mesh::new(
            geometry,
            StandardMaterial::new(Color::from_hex(color)).into(),
        ));
        o.position = Vector3::new(x, 0.75, 0.0);
        o.flicker = Some(flicker);
        scene.add(o);
    }

    // A warm light over the right of the floor, pulsing; the bulb a small
    // unlit sphere under the same flicker, as its child.
    let mut lamp = Object3D::light(PointLight::new(Color::from_hex(0xffc070), 6.0));
    lamp.position = Vector3::new(2.4, 1.1, 0.6);
    lamp.flicker = Some(Flicker::sine(2.0).range(0.1, 1.0));
    let lamp = scene.add(lamp);
    scene.add_to(
        lamp,
        Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.12, 24, 12),
            BasicMaterial::new(Color::from_hex(0xffd9a0)).into(),
        )),
    );

    camera.position = Vector3::new(2.2, 2.6, 6.4);
    // Aimed right of the objects, so they sit left of the panel.
    camera.look_at(Vector3::new(1.9, 0.5, 0.0));
}

/// A panel of labelled tiles, each flickering its own way.
fn build_ui(ui: &mut Ui) {
    let root = ui.root();
    ui.set_style(
        root,
        style! { display: flex; justify-content: flex-end; padding: 18px; },
    );
    let panel = ui.add_box(
        root,
        style! {
            display: flex; flex-direction: column; gap: 10px;
            width: 384px; padding: 16px;
            background: rgba(20, 24, 33, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 10px;
            font-family: Arial; font-size: 13px; color: #c6ccd8;
        },
    );
    let title = ui.add_box(
        panel,
        style! { font-size: 15px; font-weight: bold; color: #f2f4f8; },
    );
    ui.add_text(title, "UI nodes");

    let tiles: [(&str, u32, Flicker); 4] = [
        ("square 1 Hz", 0xe0874a, Flicker::square(1.0)),
        (
            "square 4 Hz, 25% duty",
            0x4aa3e0,
            Flicker::square(4.0).duty(0.25),
        ),
        ("sine 0.5 Hz", 0x7fd36b, Flicker::sine(0.5)),
        (
            "sine 2 Hz, 30-100%",
            0xffc070,
            Flicker::sine(2.0).range(0.3, 1.0),
        ),
    ];
    for (label, color, flicker) in tiles {
        let row = ui.add_box(
            panel,
            style! { display: flex; align-items: center; gap: 12px; },
        );
        let mut swatch = style! { width: 64px; height: 28px; border-radius: 6px; };
        swatch.background = Some(rgba(color));
        let tile = ui.add_box(row, swatch);
        ui.set_flicker(tile, Some(flicker));
        let text = ui.add_box(row, style! {});
        ui.add_text(text, label);
    }

    // Two at 10 Hz half a cycle apart: one is on while the other is off.
    let row = ui.add_box(
        panel,
        style! { display: flex; align-items: center; gap: 12px; },
    );
    for phase in [0.0, 0.5] {
        let tile = ui.add_box(
            row,
            style! { width: 26px; height: 28px; border-radius: 6px; background: #f2f4f8; },
        );
        ui.set_flicker(tile, Some(Flicker::square(10.0).phase(phase)));
    }
    let text = ui.add_box(row, style! { margin: 0 0 0 4px; });
    ui.add_text(text, "10 Hz, half a cycle apart");
}

fn rgba(hex: u32) -> [f32; 4] {
    let c = |s: u32| ((hex >> s) & 0xff) as f32 / 255.0;
    [c(16), c(8), c(0), 1.0]
}

/// Lays `ui` out over the overlay and paints it there.
fn paint_ui(ui: &mut Ui, overlay: &mut Overlay, scale: f32) {
    let (w, h) = overlay.size();
    let (w, h) = (w as f32 / scale, h as f32 / scale);
    let mut painter = OverlayPainter::with_scale(overlay, scale);
    ui.layout(w, h, &mut painter);
    ui.paint(&mut painter);
}

/// Frames a sixteenth of a second apart, headless, in a column each, to
/// `out/flicker_strip.png`.
fn strip() {
    const FRAMES: u32 = 8;
    const DT: f64 = 1.0 / 16.0;
    let mut scene = Scene::new();
    let mut camera = PerspectiveCamera::new(45.0, W as f32 / H as f32, 0.1, 100.0);
    build_scene(&mut scene, &mut camera);
    let mut ui = Ui::new();
    build_ui(&mut ui);
    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);
    let mut sheet = vec![0u8; (W * H * FRAMES * 4) as usize];
    for i in 0..FRAMES {
        let t = i as f64 * DT;
        scene.apply_flicker(t);
        let mut frame = hr.render_to_rgba(&mut scene, &camera);
        let mut overlay = Overlay::new(W, H);
        paint_ui(&mut ui, &mut overlay, 1.0);
        // The overlay is straight-alpha sRGB: over the frame, pixel by pixel.
        for (f, o) in frame.chunks_mut(4).zip(overlay.pixels().chunks(4)) {
            let a = o[3] as f32 / 255.0;
            for c in 0..3 {
                f[c] = (o[c] as f32 * a + f[c] as f32 * (1.0 - a)).round() as u8;
            }
        }
        // Frame i is rows i*H.. of the sheet: one under another.
        let start = (i * W * H * 4) as usize;
        sheet[start..start + frame.len()].copy_from_slice(&frame);
        ui.step(DT as f32);
        println!("t = {t:.4} s");
    }
    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/flicker_strip.png", encode_png(W, H * FRAMES, &sheet)).unwrap();
    println!("wrote out/flicker_strip.png");
}

fn main() {
    if std::env::args().any(|a| a == "--strip") {
        return strip();
    }
    sketch(
        SketchOptions::new("threers — flicker").size(W, H).msaa(4),
        |s| {
            build_scene(s.scene, s.camera);
            let mut ui = Ui::new();
            build_ui(&mut ui);
            move |f: &mut Frame| {
                f.scene.apply_flicker(f.elapsed as f64);
                ui.step(f.delta);
                paint_ui(&mut ui, f.overlay, f.scale_factor);
            }
        },
    );
}
