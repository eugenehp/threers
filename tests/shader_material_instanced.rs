//! A `ShaderMaterial` on an `InstancedMesh`, and per-instance custom columns.
//!
//! An instanced mesh used to draw its built-in instanced pipeline whatever its
//! material, so a custom shader on one drew nothing of its own, and there was
//! no way to hand each instance data beyond its matrix and tint. Now the
//! custom pipeline takes the instanced vertex layout, with two custom columns
//! (`InstancedMesh::set_custom`) read as `in.icustom0` and `in.icustom1`.

use threers::{
    BoxGeometry, Color, HeadlessRenderer, InstancedMesh, Material, Matrix4, Object3D,
    OrthographicCamera, Scene, ShaderMaterial, ToneMapping, Vector3,
};

const W: u32 = 128;
const H: u32 = 64;

/// Instances at x = -40, 0 and 40, each a 16-unit cube; the camera shows one
/// unit a pixel, so instance `k` covers the pixels around (64 + 40 (k - 1), 32).
fn render(mut im: InstancedMesh) -> Option<Vec<u8>> {
    let mut r = HeadlessRenderer::builder()
        .size(W, H)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()?;
    r.renderer().set_tone_mapping(ToneMapping::None, 1.0);
    for (k, t) in im.transforms.iter_mut().enumerate() {
        *t = Matrix4::translation(Vector3::new(40.0 * (k as f32 - 1.0), 0.0, 0.0));
    }
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::instanced_mesh(im));
    let mut cam = OrthographicCamera::new(-64.0, 64.0, 32.0, -32.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 50.0);
    scene.update_world();
    Some(r.render_to_rgba(&mut scene, &cam))
}

fn at(px: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

const VERTEX_COLOR: &str = "@fragment\n\
     fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
     return vec4<f32>(in.vertex_color.rgb, 1.0);\n\
     }\n";

fn cubes(material: ShaderMaterial) -> InstancedMesh {
    let mut g = BoxGeometry::new(16.0, 16.0, 16.0);
    // White vertex colours, so the tint and the custom colour come through.
    let n = g.get_attribute("position").unwrap().count();
    g.set_attribute("color", threers::BufferAttribute::new(vec![1.0; n * 4], 4));
    InstancedMesh::new(g, Material::Shader(material), 3)
}

/// Without a vertex stage of its own the material still draws every instance
/// at its matrix, tinted, through the built-in instanced entry.
#[test]
fn a_shader_material_draws_every_instance() {
    let mut im = cubes(ShaderMaterial::new(VERTEX_COLOR));
    im.colors = vec![
        Color::from_hex(0xff0000),
        Color::from_hex(0x00ff00),
        Color::from_hex(0x0000ff),
    ];
    let Some(px) = render(im) else { return };
    assert_eq!(at(&px, 24, 32), [255, 0, 0], "instance 0 at x = -40");
    assert_eq!(at(&px, 64, 32), [0, 255, 0], "instance 1 at x = 0");
    assert_eq!(at(&px, 104, 32), [0, 0, 255], "instance 2 at x = 40");
    assert_eq!(at(&px, 44, 32), [0, 0, 0], "nothing between them");
}

/// A custom vertex stage reads both custom columns: one moves each instance
/// up, the other colours it.
#[test]
fn custom_columns_reach_a_custom_vertex_stage() {
    let material = ShaderMaterial::new(VERTEX_COLOR).with_vertex(
        "@vertex\n\
         fn vs_custom(in: InstancedCustomIn, @builtin(vertex_index) vid: u32,\n\
                      @builtin(instance_index) iid: u32) -> VsOut {\n\
         var out = standard_vertex_instanced(instanced_in(in), vid, iid);\n\
         out.world_pos.y += in.icustom0.x;\n\
         out.clip_pos = frame.view_proj * vec4<f32>(out.world_pos, 1.0);\n\
         out.vertex_color = vec4<f32>(in.icustom1.rgb, 1.0);\n\
         return out;\n\
         }\n",
    );
    let mut im = cubes(material);
    // Heights: instance 1 rises 20; instance 2 has no row and stays put.
    im.set_custom(0, 1, &[0.0, 20.0]);
    im.set_custom(1, 3, &[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]);
    let Some(px) = render(im) else { return };
    assert_eq!(
        at(&px, 24, 32),
        [255, 255, 0],
        "instance 0: yellow, in place"
    );
    assert_eq!(
        at(&px, 64, 32 - 20),
        [0, 255, 255],
        "instance 1: cyan, 20 up"
    );
    assert_eq!(at(&px, 64, 32 + 6), [0, 0, 0], "and gone from where it was");
    assert_eq!(
        at(&px, 104, 32),
        [255, 0, 255],
        "instance 2: magenta, in place"
    );
}

/// The same material on a plain mesh and on an instanced one gets two
/// pipelines, one of each layout, and both draw in the same frame.
#[test]
fn one_material_on_a_mesh_and_an_instanced_mesh() {
    let Ok(mut r) = HeadlessRenderer::builder()
        .size(W, H)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
    else {
        return;
    };
    r.renderer().set_tone_mapping(ToneMapping::None, 1.0);
    let material = ShaderMaterial::new(VERTEX_COLOR);
    let mut im = cubes(material.clone());
    im.transforms.truncate(1);
    im.transforms[0] = Matrix4::translation(Vector3::new(-40.0, 0.0, 0.0));
    im.colors = vec![Color::from_hex(0xff0000)];
    let mut g = BoxGeometry::new(16.0, 16.0, 16.0);
    let n = g.get_attribute("position").unwrap().count();
    g.set_attribute("color", threers::BufferAttribute::new(vec![1.0; n * 4], 4));
    let mut mesh = Object3D::mesh(threers::Mesh::new(g, Material::Shader(material)));
    mesh.position = Vector3::new(40.0, 0.0, 0.0);
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::instanced_mesh(im));
    scene.add(mesh);
    let mut cam = OrthographicCamera::new(-64.0, 64.0, 32.0, -32.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 50.0);
    scene.update_world();
    let px = r.render_to_rgba(&mut scene, &cam);
    assert_eq!(at(&px, 24, 32), [255, 0, 0], "the instance, tinted");
    assert_eq!(at(&px, 104, 32), [255, 255, 255], "the plain mesh");
}

/// The instance buffer is kept between frames now; changing the data must
/// still show, and leaving it alone must keep showing the same thing.
#[test]
fn instance_data_changed_between_frames_is_drawn() {
    let Ok(mut r) = HeadlessRenderer::builder()
        .size(W, H)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
    else {
        return;
    };
    r.renderer().set_tone_mapping(ToneMapping::None, 1.0);
    let mut im = cubes(ShaderMaterial::new(VERTEX_COLOR));
    for (k, t) in im.transforms.iter_mut().enumerate() {
        *t = Matrix4::translation(Vector3::new(40.0 * (k as f32 - 1.0), 0.0, 0.0));
    }
    im.colors = vec![Color::from_hex(0xff0000); 3];
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let id = scene.add(Object3D::instanced_mesh(im));
    let mut cam = OrthographicCamera::new(-64.0, 64.0, 32.0, -32.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 50.0);
    scene.update_world();
    let px = r.render_to_rgba(&mut scene, &cam);
    assert_eq!(at(&px, 104, 32), [255, 0, 0]);
    // Same data again: same picture.
    let again = r.render_to_rgba(&mut scene, &cam);
    assert_eq!(px, again);
    // Move instance 2 up and recolour instance 0.
    if let Some(threers::core::ObjectKind::InstancedMesh(im)) = scene.arena.get_mut(id).map(|o| &mut o.kind) {
        im.transforms[2] = Matrix4::translation(Vector3::new(40.0, 20.0, 0.0));
        im.colors[0] = Color::from_hex(0x00ff00);
    }
    let moved = r.render_to_rgba(&mut scene, &cam);
    assert_eq!(at(&moved, 104, 32), [0, 0, 0], "instance 2 left its old place");
    assert_eq!(at(&moved, 104, 12), [255, 0, 0], "and is drawn at its new one");
    assert_eq!(at(&moved, 24, 32), [0, 255, 0], "instance 0 recoloured");
}

/// A timing, not a check: 200k still instances a frame, with
/// `VERSIONED=1` declaring an `instance_version`. Run with `--ignored
/// --nocapture`. On an M4 Pro: 8.6 ms rebuilding every frame, 7.3 with the
/// kept buffer, 4.0 versioned.
#[test]
#[ignore]
fn time_static_instances() {
    let Ok(mut r) = HeadlessRenderer::builder().size(256, 256).build() else { return };
    let mut im = InstancedMesh::new(BoxGeometry::new(0.5, 0.5, 0.5), Material::Basic(Default::default()), 200_000);
    for (k, t) in im.transforms.iter_mut().enumerate() {
        *t = Matrix4::translation(Vector3::new((k % 500) as f32 - 250.0, (k / 500) as f32 - 200.0, 0.0));
    }
    if std::env::var_os("VERSIONED").is_some() {
        im.instance_version = Some(1);
    }
    let mut scene = Scene::new();
    scene.add(Object3D::instanced_mesh(im));
    let mut cam = OrthographicCamera::new(-260.0, 260.0, 210.0, -210.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 50.0);
    scene.update_world();
    let _ = r.render_to_rgba(&mut scene, &cam);
    let t = std::time::Instant::now();
    for _ in 0..30 { let _ = r.render_to_rgba(&mut scene, &cam); }
    eprintln!("200k static instances: {:.2} ms/frame", t.elapsed().as_secs_f64() * 1e3 / 30.0);
}

/// With an `instance_version`, the data is read only when the version moves:
/// an edit without a bump is not shown (that is the point), and a bump shows
/// it.
#[test]
fn a_versioned_instanced_mesh_reads_its_data_when_the_version_moves() {
    let Ok(mut r) = HeadlessRenderer::builder()
        .size(W, H)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
    else {
        return;
    };
    r.renderer().set_tone_mapping(ToneMapping::None, 1.0);
    let mut im = cubes(ShaderMaterial::new(VERTEX_COLOR));
    for (k, t) in im.transforms.iter_mut().enumerate() {
        *t = Matrix4::translation(Vector3::new(40.0 * (k as f32 - 1.0), 0.0, 0.0));
    }
    im.colors = vec![Color::from_hex(0xff0000); 3];
    im.instance_version = Some(1);
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let id = scene.add(Object3D::instanced_mesh(im));
    let mut cam = OrthographicCamera::new(-64.0, 64.0, 32.0, -32.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 50.0);
    scene.update_world();
    let _ = r.render_to_rgba(&mut scene, &cam);
    let edit = |scene: &mut Scene, bump: bool| {
        if let Some(threers::core::ObjectKind::InstancedMesh(im)) =
            scene.arena.get_mut(id).map(|o| &mut o.kind)
        {
            im.colors[0] = Color::from_hex(0x00ff00);
            if bump {
                im.instance_version = Some(2);
            }
        }
    };
    edit(&mut scene, false);
    let px = r.render_to_rgba(&mut scene, &cam);
    assert_eq!(at(&px, 24, 32), [255, 0, 0], "not re-read without a bump");
    edit(&mut scene, true);
    let px = r.render_to_rgba(&mut scene, &cam);
    assert_eq!(at(&px, 24, 32), [0, 255, 0], "re-read once the version moved");
}
