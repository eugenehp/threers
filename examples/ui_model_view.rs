//! A threers scene inside a UI card: `ModelView` renders a torus knot with
//! the GPU renderer at the card's size into an image node, and the card is
//! painted with the rest of the UI. The orbit is turned (as a drag would)
//! and the card drawn again. Writes `out/ui_model_view.png`: before and
//! after, side by side.
//!
//! ```bash
//! cargo run --release --features ui --example ui_model_view
//! ```

use threers::lights::Light;
use threers::prelude::*;
use threers::style;
use threers::ui::model3d::Orbit;
use threers::ui::model_view::ModelView;
use threers::ui::{paint_to_rgba, FixedWidthText, Ui};
use threers::utils::png::encode_png;

const W: u32 = 420;
const H: u32 = 360;

fn main() {
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(
        root,
        style! { background: #12151b; padding: 20px; display: flex; flex-direction: column; },
    );
    let card = ui.add_box(root, style! { display: flex; flex-direction: column; gap: 10px; padding: 14px; border-radius: 14px; background: #1f2430; color: #e5e7eb; font-size: 14px; });
    let title = ui.add_box(card, style! { font-weight: bold; font-size: 16px; });
    ui.add_text(title, "Torus knot");
    let mut view = ModelView::new(
        &mut ui,
        card,
        style! { width: 352px; height: 240px; border-radius: 10px; },
    );
    let caption = ui.add_box(card, style! { font-size: 12px; color: rgb(151, 158, 175); });
    ui.add_text(caption, "Drag to turn it, as you would in the app.");
    ui.layout(W as f32, H as f32, &mut FixedWidthText(7.0));

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0f1218);
    scene.environment = Some(threers::extras::RoomEnvironment::pmrem(128));
    scene.environment_intensity = 0.5;
    scene.add_light(AmbientLight::new(Color::WHITE, 0.25));
    let mut l = Object3D::light(Light::Directional(DirectionalLight::new(
        Color::from_hex(0xfff0e0),
        2.4,
    )));
    l.position = Vector3::new(-3.0, 5.0, 4.0);
    scene.add(l);
    let mut m = StandardMaterial::new(Color::from_hex(0x2563eb));
    m.roughness = 0.35;
    m.metalness = 0.2;
    scene.add(Object3D::mesh(Mesh::new(
        TorusKnotGeometry::new(0.8, 0.26, 220, 32, 2, 3),
        m.into(),
    )));

    let target = Vector3::new(0.0, 0.0, 0.0);
    let mut orbit = Orbit {
        distance: 4.6,
        ..Orbit::default()
    };
    let mut frames = vec![];
    for _ in 0..2 {
        view.render(&mut ui, &mut scene, &orbit, target, 1.0)
            .expect("a GPU adapter");
        frames.push(paint_to_rgba(&ui, 1.0, W, H));
        // A drag to the right, and down a little.
        orbit = orbit.turned(1.1, 0.35);
    }
    // Side by side.
    let mut out = vec![0u8; (W * 2 * H * 4) as usize];
    for y in 0..H as usize {
        for (k, f) in frames.iter().enumerate() {
            let row = &f[y * W as usize * 4..(y + 1) * W as usize * 4];
            let at = (y * W as usize * 2 + k * W as usize) * 4;
            out[at..at + row.len()].copy_from_slice(row);
        }
    }
    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/ui_model_view.png", encode_png(W * 2, H, &out)).unwrap();
    println!("wrote out/ui_model_view.png");
}
