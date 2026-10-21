//! A screen-space shader material (`ShaderMaterial::with_screen_space`) reads
//! the opaque scene behind it: `ss_color_tex` and `ss_depth_tex` at group 3.
//!
//! A red box behind a pane that covers the view; the pane swaps the capture's
//! red and blue and marks in green where the capture's depth found geometry.
//! So the box shows blue and green through it, and the background neither.
//!
//! This went unnoticed once: the capture moved out of the main layout's
//! group 3, every screen-space shader material failed to build its pipeline,
//! and the ocean example drew no water. Skipped when no GPU adapter is
//! available.

use threers::prelude::*;

const N: u32 = 64;

const PANE: &str = r#"
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(ss_depth_tex));
    let uv = in.clip_pos.xy / dims;
    let c = textureSampleLevel(ss_color_tex, ss_color_samp, uv, 0.0).rgb;
    let raw = textureLoad(ss_depth_tex, vec2<i32>(in.clip_pos.xy), 0).r;
    let d = select(raw, 1.0 - raw, frame.camera_position.w > 0.5);
    let hit = select(0.0, 1.0, d < 0.9999);
    return vec4<f32>(c.b, hit, c.r, 1.0);
}
"#;

#[test]
fn a_screen_space_shader_reads_the_scene_behind_it() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipped: no GPU adapter");
        return;
    };
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let mut red = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        Material::Basic(BasicMaterial::new(Color::new(1.0, 0.0, 0.0))),
    ));
    red.position = Vector3::new(0.0, 0.0, -2.0);
    scene.add(red);
    let pane = ShaderMaterial::new(PANE).with_screen_space(true);
    let mut pane = Object3D::mesh(Mesh::new(PlaneGeometry::new(4.0, 4.0), pane.into()));
    pane.position = Vector3::new(0.0, 0.0, 1.0);
    scene.add(pane);
    let mut camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 50.0);
    camera.position = Vector3::new(0.0, 0.0, 3.0);
    camera.look_at(Vector3::new(0.0, 0.0, -2.0));

    let px = hr.render_to_rgba(&mut scene, &camera);
    let at = |x: u32, y: u32| {
        let i = ((y * N + x) * 4) as usize;
        [px[i], px[i + 1], px[i + 2]]
    };
    let centre = at(N / 2, N / 2);
    let corner = at(2, 2);
    eprintln!("centre {centre:?}, corner {corner:?}");
    assert!(
        centre[2] > 200 && centre[1] > 200 && centre[0] < 40,
        "the box does not show through the pane as blue and green: {centre:?}"
    );
    assert!(
        corner.iter().all(|&c| c < 40),
        "the background shows through as something: {corner:?}"
    );
}
