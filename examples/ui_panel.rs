//! `threers::ui`: a control panel written in CSS terms — flexbox, padding,
//! borders, rounded corners — with Chrome's own form controls (range,
//! checkbox, `<select>`, colour well, button), laid out, hit-tested and
//! painted over a 3D render through `OverlayPainter`. A click is simulated
//! on the roughness slider: `hit_test` finds the slider under the pointer,
//! `range_fraction` turns the pointer into a value, and the material the
//! panel drives is drawn with it. Written to `out/ui_panel.png`.
//!
//! ```bash
//! cargo run --release --features ui --example ui_panel
//! ```

use threers::lights::Light;
use threers::overlay::Overlay;
use threers::prelude::*;
use threers::style;
use threers::ui::{controls, Edges, NodeId, OverlayPainter, Style, TextMeasure, Ui};

const W: u32 = 960;
const H: u32 = 540;

/// The panel's settings, as the controls show them.
struct Settings {
    roughness: f32,
    metalness: f32,
    color: [f32; 3],
    shadows: bool,
    spin: bool,
    tone: &'static str,
}

/// The panel's interactive nodes, for hit-testing.
struct Panel {
    roughness: NodeId,
}

/// A labelled row: the label on the left, the control on the right.
fn row(ui: &mut Ui, parent: NodeId, label: &str, font: &Style) -> NodeId {
    let r = ui.add_box(
        parent,
        style! { display: flex; align-items: center; justify-content: space-between; margin: 0 0 12px; },
    );
    let l = ui.add_box(r, font.clone());
    ui.add_text(l, label);
    r
}

fn build(ui: &mut Ui, s: &Settings, text: &mut dyn TextMeasure) -> Panel {
    let root = ui.root();
    ui.set_style(
        root,
        style! { display: flex; justify-content: flex-end; align-items: flex-start; padding: 18px; },
    );
    let card = ui.add_box(
        root,
        style! {
            display: flex;
            flex-direction: column;
            width: 280px;
            padding: 16px 18px 6px;
            background: rgba(20, 24, 33, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 10px;
        },
    );
    let title = ui.add_box(card, style! { font-family: Arial; font-size: 15px; font-weight: bold; color: #f2f4f8; margin: 0 0 14px; });
    ui.add_text(title, "Material");
    let font = style! { font-family: Arial; font-size: 13px; color: #c6ccd8; };
    let control_font = style! { font-family: Arial; font-size: 13px; color: black; };

    let r = row(ui, card, "Roughness", &font);
    let roughness = controls::range(ui, r, s.roughness, true);
    let r = row(ui, card, "Metalness", &font);
    controls::range(ui, r, s.metalness, false);
    let r = row(ui, card, "Colour", &font);
    controls::color_well(ui, r, s.color, 60.0);
    let r = row(ui, card, "Tone mapping", &font);
    controls::select(
        ui,
        r,
        s.tone,
        &["None", "ACES", "AgX", "Neutral"],
        &control_font,
        text,
    );
    let r = row(ui, card, "Cast shadows", &font);
    controls::checkbox(ui, r, s.shadows, 13.0, controls::BORDER, Edges::default());
    let r = row(ui, card, "Spin", &font);
    controls::checkbox(ui, r, s.spin, 13.0, controls::BORDER, Edges::default());
    let r = ui.add_box(
        card,
        style! { display: flex; justify-content: flex-end; margin: 4px 0 12px; },
    );
    controls::button(ui, r, "Export glTF", &control_font, text);
    Panel { roughness }
}

fn main() {
    let mut s = Settings {
        roughness: 0.7,
        metalness: 0.0,
        color: [0.85, 0.32, 0.18],
        shadows: true,
        spin: false,
        tone: "ACES",
    };

    // Lay the panel out once to find the slider, then "click" two thirds
    // of the way along it: the hit test says what is under the pointer, and
    // the slider's rectangle says what value that is.
    let mut overlay = Overlay::new(W, H);
    let mut ui = Ui::new();
    let panel = {
        let mut measure = OverlayPainter::new(&mut overlay);
        let panel = build(&mut ui, &s, &mut measure);
        ui.layout(W as f32, H as f32, &mut measure);
        panel
    };
    let slider = ui.rect(panel.roughness);
    let (px, py) = (
        slider.x + 8.0 + (slider.width - 16.0) * 0.25,
        slider.y + slider.height / 2.0,
    );
    let hit = ui.hit_test(px, py).expect("something under the pointer");
    assert!(
        ui.ancestors(hit).any(|a| a == panel.roughness) || hit == panel.roughness,
        "the click landed on the roughness slider"
    );
    s.roughness = controls::range_fraction(slider, px);
    println!("click at ({px:.0}, {py:.0}) → roughness {:.2}", s.roughness);

    // The panel again, showing the new value, painted onto the overlay.
    let mut ui = Ui::new();
    {
        let mut painter = OverlayPainter::new(&mut overlay);
        build(&mut ui, &s, &mut painter);
        ui.layout(W as f32, H as f32, &mut painter);
        ui.paint(&mut painter);
    }

    // The scene the panel drives.
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0e1117);
    scene.environment = Some(threers::extras::RoomEnvironment::pmrem(128));
    scene.environment_intensity = 0.5;
    scene.add_light(AmbientLight::new(Color::WHITE, 0.2));
    let mut sun = DirectionalLight::new(Color::from_hex(0xfff0e0), 2.4);
    sun.cast_shadow = s.shadows;
    sun.shadow.camera_size = 6.0;
    sun.shadow.normal_bias = 0.02;
    let mut l = Object3D::light(Light::Directional(sun));
    l.position = Vector3::new(-3.0, 6.0, 4.0);
    scene.add(l);
    let mut floor_m = StandardMaterial::new(Color::from_hex(0x2a303c));
    floor_m.roughness = 0.9;
    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(30.0, 30.0), floor_m.into()));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    scene.add(floor);
    // The well shows the colour as CSS does, sRGB-encoded; a material's
    // colour is linear.
    let mut m = StandardMaterial::new(Color::from_srgb(s.color));
    m.roughness = s.roughness;
    m.metalness = s.metalness;
    let mut knot = Object3D::mesh(Mesh::new(
        TorusKnotGeometry::new(0.8, 0.26, 220, 32, 2, 3),
        m.into(),
    ));
    knot.position = Vector3::new(0.0, 1.35, 0.0);
    knot.cast_shadow = s.shadows;
    scene.add(knot);

    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);
    hr.renderer().set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
    let mut camera = PerspectiveCamera::new(40.0, W as f32 / H as f32, 0.1, 50.0);
    camera.position = Vector3::new(1.6, 2.6, 6.2);
    // Aimed right of the knot, so it sits in the space the panel leaves.
    camera.look_at(Vector3::new(1.3, 1.1, 0.0));
    let mut frame = hr.render_to_rgba(&mut scene, &camera);

    // The overlay is straight-alpha sRGB: over the frame, pixel by pixel.
    for (f, o) in frame.chunks_mut(4).zip(overlay.pixels().chunks(4)) {
        let a = o[3] as f32 / 255.0;
        for c in 0..3 {
            f[c] = (o[c] as f32 * a + f[c] as f32 * (1.0 - a)).round() as u8;
        }
    }
    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/ui_panel.png", encode_png(W, H, &frame)).unwrap();
    println!("wrote out/ui_panel.png");
}
