//! CAD-style quad view: top, front and side orthographic panes plus a
//! perspective pane, in one image, rendered headless to a PNG.
//!
//! Layout is the third-angle drawing sheet — top view above the front view,
//! right-side view beside it — so features line up across the orthographic
//! panes, which share one scale. Pass `--max front|top|side|persp` to render a
//! single maximised pane instead.
//!
//! Every pane is drawn with 4× MSAA (through a pane-sized multisampled target),
//! and the directional light's shadow map is drawn once for all four panes.
//!
//! ```text
//! cargo run --release --example quad_view [-- out/quad_view.png] [--max front]
//! ```

use threers::helpers::{GridHelper, Pane, QuadView};
use threers::{
    encode_png, AmbientLight, BoxGeometry, Color, CylinderGeometry, DirectionalLight,
    HeadlessRenderer, Material, Mesh, Object3D, PerspectiveCamera, Scene, StandardMaterial,
    TorusGeometry, Vector3,
};

fn part(scene: &mut Scene) {
    let steel = |hex: u32| {
        Material::Standard(
            StandardMaterial::new(Color::from_hex(hex))
                .with_roughness(0.45)
                .with_metalness(0.2),
        )
    };
    // A bracket: base plate, an upright, a boss on top and a ring on the side.
    let mut base = Object3D::mesh(Mesh::new(BoxGeometry::new(4.0, 0.4, 2.4), steel(0x8aa1b4)));
    base.position = Vector3::new(0.0, 0.2, 0.0);
    base.receive_shadow = true;
    scene.add(base);
    let mut upright = Object3D::mesh(Mesh::new(BoxGeometry::new(0.5, 2.0, 2.4), steel(0x6f8799)));
    upright.position = Vector3::new(-1.75, 1.4, 0.0);
    upright.cast_shadow = true;
    upright.receive_shadow = true;
    scene.add(upright);
    let mut boss = Object3D::mesh(Mesh::new(
        CylinderGeometry::new(0.55, 0.55, 1.0, 48, 1, false, 0.0, std::f32::consts::TAU),
        steel(0xd08a3a),
    ));
    boss.position = Vector3::new(0.8, 0.9, 0.0);
    boss.cast_shadow = true;
    boss.receive_shadow = true;
    scene.add(boss);
    let mut ring = Object3D::mesh(Mesh::new(
        TorusGeometry::new(0.45, 0.12, 24, 64, std::f32::consts::TAU),
        steel(0x3a9a6a),
    ));
    ring.position = Vector3::new(-1.45, 1.6, 0.0);
    ring.rotate_y(std::f32::consts::FRAC_PI_2);
    ring.cast_shadow = true;
    scene.add(ring);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut out = "out/quad_view.png".to_string();
    let mut maximized = None;
    while let Some(a) = args.next() {
        if a == "--max" {
            let name = args.next();
            maximized = match name.as_deref().and_then(Pane::from_name) {
                Some(pane) => Some(pane),
                None => {
                    eprintln!("--max takes top|front|side|persp, got {name:?}");
                    std::process::exit(2);
                }
            };
        } else {
            out = a;
        }
    }

    let (w, h) = (1200u32, 900u32);
    let mut hr = match HeadlessRenderer::builder().size(w, h).build() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — needs a GPU adapter.");
            std::process::exit(2);
        }
    };

    hr.set_msaa(4);

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x2a3138);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.55));
    let mut sun = DirectionalLight::new(Color::WHITE, 2.4)
        .with_direction(Vector3::new(-0.4, -0.8, -0.45).normalize());
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow.camera_size = 4.0;
    scene.add_light(sun);
    part(&mut scene);

    let mut quad = QuadView::new();
    // Frame the part before the grid goes in, so the grid does not set the
    // scale of the orthographic panes.
    quad.frame_scene(&mut scene).expect("the part has bounds");
    scene.add(GridHelper::new(
        8.0,
        16,
        Color::from_hex(0x56606a),
        Color::from_hex(0x343b42),
    ));

    quad.gap = 4.0;
    quad.gap_color = Some(Color::from_hex(0x0c0e10));
    // Ortho panes a shade cooler than the perspective one, as CAD tools do.
    let ortho_bg = Color::from_hex(0x1d2329);
    quad.backgrounds = [Some(ortho_bg), Some(ortho_bg), Some(ortho_bg), None];
    quad.maximized = maximized;

    let mut persp = PerspectiveCamera::new(40.0, 1.0, 0.1, 100.0);
    persp.position = Vector3::new(6.0, 4.5, 7.0);
    persp.look_at(quad.target());

    quad.render(&mut hr, &mut scene, &persp);
    let rgba = hr.read_rgba();

    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, encode_png(w, h, &rgba)).expect("write png");
    println!("wrote {out} ({w}x{h})");
}
