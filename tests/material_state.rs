//! three.js's `depthTest` / `depthWrite` / `depthFunc` / `colorWrite` on the
//! wgpu renderer, for every kind of draw. The scenes and the expected pixels
//! are in `support/material_state.rs`; `tests/backend_parity.rs` holds the
//! Metal backend to the same ones.

#[path = "support/material_state.rs"]
mod state;

use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::materials::{Material, RenderState, ShaderMaterial};
use threers::math::{Color, Vector3};

#[test]
fn material_depth_and_colour_state_on_every_draw_kind() {
    let mut failures = Vec::new();
    for format in [wgpu::TextureFormat::Rgba8UnormSrgb, wgpu::TextureFormat::Rgba8Unorm] {
        for msaa in [1, 4] {
            let Ok(mut r) = threers::HeadlessRenderer::builder()
                .size(state::N, state::N)
                .color_format(format)
                .build()
            else {
                eprintln!("skipping: no GPU adapter");
                return;
            };
            r.set_msaa(msaa);
            failures.extend(state::check(
                &format!("wgpu {format:?} msaa {msaa}"),
                &state::KINDS,
                &mut |s, c| r.render_to_rgba(s, c),
            ));
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// The same state on a `ShaderMaterial`, opaque and transparent: a custom
/// pipeline is compiled per shader, and the state is applied over it.
#[test]
fn shader_material_takes_the_material_state() {
    const FRAG: &str = "@fragment\nfn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n    let c = u_data.data[0];\n    return vec4<f32>(framebuffer_encode(c.rgb), c.a);\n}\n";
    let Ok(mut r) = threers::HeadlessRenderer::builder().size(state::N, state::N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut failures = Vec::new();
    for transparent in [false, true] {
        for case in state::cases() {
            let mut scene = threers::scene::Scene::new();
            scene.background = Color::from_hex(state::BACKGROUND);
            let mut wm = threers::materials::BasicMaterial::new(Color::from_hex(state::WALL));
            wm.render_state = case.wall;
            let mut wall = Object3D::mesh(Mesh::new(PlaneGeometry::new(2.5, 2.5), wm.into()));
            wall.position = Vector3::new(0.0, 0.0, 0.5);
            scene.add(wall);
            let green = Color::from_hex(state::SUBJECT);
            let mut m = ShaderMaterial::new(FRAG).with_data(vec![[green.r, green.g, green.b, 1.0]]);
            m.transparent = transparent;
            m.render_state = case.subject;
            let mut s = Object3D::mesh(Mesh::new(PlaneGeometry::new(2.5, 2.5), Material::Shader(m)));
            s.position = Vector3::new(0.0, 0.0, -0.5);
            s.render_order = 1;
            scene.add(s);
            let rgba = r.render_to_rgba(&mut scene, &state::camera());
            let (got, px) = state::classify(&rgba);
            eprintln!("shader transparent={transparent} {}: {got:06x} {px:?}", case.name);
            if got != case.expect {
                failures.push(format!(
                    "shader (transparent {transparent}) — {}: centre {px:?}, want {:06x}",
                    case.name, case.expect
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// three.js writes depth for a transparent material unless `depthWrite` is
/// false. With the Rust default (`depth_write: None`) a blended surface does
/// not write; with `Some(true)` — what the JS layer sends for three.js's
/// default — it hides an opaque-ordered subject drawn after it.
#[test]
fn a_transparent_material_writes_depth_when_asked() {
    let Ok(mut r) = threers::HeadlessRenderer::builder().size(state::N, state::N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut failures = Vec::new();
    for (write, expect_subject) in [(None, true), (Some(true), false)] {
        let mut scene = threers::scene::Scene::new();
        scene.background = Color::from_hex(state::BACKGROUND);
        let mut wm = threers::materials::BasicMaterial::new(Color::from_hex(state::WALL));
        wm.transparent = true;
        wm.opacity = 0.999;
        wm.render_state = RenderState { depth_write: write, ..RenderState::default() };
        let mut wall = Object3D::mesh(Mesh::new(PlaneGeometry::new(2.5, 2.5), wm.into()));
        wall.position = Vector3::new(0.0, 0.0, 0.5);
        scene.add(wall);
        // Also transparent, sorted after the wall by render order, with the
        // test on: it shows only if the wall left the depth buffer alone.
        let mut sm = threers::materials::BasicMaterial::new(Color::from_hex(state::SUBJECT));
        sm.transparent = true;
        sm.opacity = 0.999;
        let mut subject = Object3D::mesh(Mesh::new(PlaneGeometry::new(2.5, 2.5), sm.into()));
        subject.position = Vector3::new(0.0, 0.0, -0.5);
        subject.render_order = 1;
        scene.add(subject);
        let rgba = r.render_to_rgba(&mut scene, &state::camera());
        let (got, px) = state::classify(&rgba);
        eprintln!("transparent wall depth_write {write:?}: {got:06x} {px:?}");
        let want = if expect_subject { state::SUBJECT } else { state::WALL };
        if got != want {
            failures.push(format!("depth_write {write:?}: centre {px:?}, want {want:06x}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
