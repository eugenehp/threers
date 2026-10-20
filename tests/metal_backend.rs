//! End-to-end tests for the Metal backend: every one of these runs a real GPU
//! and inspects the pixels that come back.
//!
//! Requires a Metal device, so the whole file is skipped when the `metal`
//! feature is off or the target is not Apple. Where a machine has the feature
//! on but no GPU access at all (a sandbox, some CI containers), `MetalDevice`
//! returns [`MetalError::NoDevice`] and the tests report themselves skipped
//! rather than failing — a missing GPU is not a broken backend.

#![cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]


use threers::cameras::{Camera, OrthographicCamera};
use threers::core::{
    BufferAttribute, BufferGeometry, InstancedMesh, LineSegments, Mesh, Object3D, Points,
};
use threers::geometries::{BoxGeometry, PlaneGeometry};
use threers::lights::{AmbientLight, DirectionalLight};
use threers::materials::{
    BasicMaterial, LineBasicMaterial, Material, PointsMaterial, StandardMaterial,
};
use threers::math::{Color, Matrix4, Vector3};
use threers::metal::{
    MetalDevice, MetalError, MetalHeadlessRenderer, MetalRenderTarget, RenderView,
};
use threers::scene::Scene;
use threers::textures::{Texture, TextureFilter, TextureFormat};

const SIZE: u32 = 64;

/// A renderer, or `None` when this machine has no Metal device — in which case
/// the caller returns early and the test passes vacuously.
fn renderer(msaa: u32) -> Option<MetalHeadlessRenderer> {
    match MetalHeadlessRenderer::builder()
        .size(SIZE, SIZE)
        .msaa(msaa)
        .build()
    {
        Ok(r) => Some(r),
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            None
        }
        Err(e) => panic!("Metal renderer could not be built: {e}"),
    }
}

/// An orthographic camera whose frustum is exactly the unit square, so a
/// `PlaneGeometry(2, 2)` at the origin fills the frame.
fn unit_camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 2.0);
    cam.target = Vector3::ZERO;
    cam
}

/// The RGBA at a fractional position in the image, `(0, 0)` top-left.
fn pixel_at(rgba: &[u8], fx: f32, fy: f32) -> [u8; 4] {
    let x = ((fx * SIZE as f32) as u32).min(SIZE - 1) as usize;
    let y = ((fy * SIZE as f32) as u32).min(SIZE - 1) as usize;
    let i = (y * SIZE as usize + x) * 4;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

fn center(rgba: &[u8]) -> [u8; 4] {
    pixel_at(rgba, 0.5, 0.5)
}

fn close(a: [u8; 4], b: [u8; 4], tolerance: i32) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (*x as i32 - *y as i32).abs() <= tolerance)
}

/// A `PlaneGeometry(2, 2)` mesh at the origin, filling [`unit_camera`]'s frame.
fn full_quad(material: Material) -> Object3D {
    Object3D::mesh(Mesh::new(PlaneGeometry::new(2.0, 2.0), material))
}

#[test]
fn device_reports_a_name_and_a_library() {
    let Some(hr) = renderer(1) else { return };
    let device = hr.device();
    assert!(!device.name().is_empty(), "device has no name");
    for entry in ["vs_mesh", "fs_mesh", "vs_point", "fs_point"] {
        assert!(
            device.function(entry).is_ok(),
            "shader library is missing `{entry}`"
        );
    }
    match device.function("no_such_function") {
        Err(MetalError::MissingFunction(name)) => assert_eq!(name, "no_such_function"),
        other => panic!("expected MissingFunction, got {other:?}"),
    }
}

#[test]
fn shader_compilation_errors_carry_the_diagnostics() {
    if MetalDevice::new().is_err() {
        return;
    }
    let err = MetalDevice::with_shader_source("this is not Metal Shading Language")
        .expect_err("nonsense MSL must not compile");
    match err {
        MetalError::ShaderCompilation(message) => {
            assert!(
                !message.is_empty(),
                "the compiler's diagnostics were thrown away"
            );
        }
        other => panic!("expected ShaderCompilation, got {other:?}"),
    }
}

#[test]
fn background_clears_the_frame() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();

    assert_eq!(rgba.len(), (SIZE * SIZE * 4) as usize);
    assert!(
        rgba.chunks_exact(4).all(|px| px == [0, 0, 0, 255]),
        "an empty scene should be nothing but the clear colour"
    );

    // A mid grey round-trips through the sRGB attachment to the byte it came from.
    scene.background = Color::from_hex(0x808080);
    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    assert!(
        close(center(&rgba), [0x80, 0x80, 0x80, 255], 2),
        "clear colour did not round-trip: {:?}",
        center(&rgba)
    );
}

#[test]
fn an_unlit_quad_covers_the_frame() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(full_quad(Material::Basic(BasicMaterial::new(
        Color::from_hex(0xff0000),
    ))));

    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    for (fx, fy) in [(0.5, 0.5), (0.1, 0.1), (0.9, 0.9)] {
        assert!(
            close(pixel_at(&rgba, fx, fy), [255, 0, 0, 255], 2),
            "quad missing at ({fx}, {fy}): {:?}",
            pixel_at(&rgba, fx, fy)
        );
    }
    let stats = hr.stats();
    assert_eq!(stats.draw_calls, 1);
    assert_eq!(stats.triangles, 2);
}

#[test]
fn a_lit_cube_is_shaded_by_its_lights() {
    let Some(mut hr) = renderer(4) else { return };
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        Material::Standard(StandardMaterial::new(Color::from_hex(0xffffff))),
    )));
    scene.add_light(AmbientLight::new(Color::WHITE, 0.2));
    // Pointing down the -Z axis, straight at the face the camera sees.
    let mut light = Object3D::light(DirectionalLight::new(Color::WHITE, 1.0));
    light.position = Vector3::new(0.0, 0.0, 5.0);
    scene.add(light);

    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    let lit = center(&rgba);
    assert!(lit[0] > 60, "the lit face is too dark: {lit:?}");
    assert_eq!(
        pixel_at(&rgba, 0.02, 0.02),
        [0, 0, 0, 255],
        "corner is not background"
    );

    // Same scene without the lights: ambient alone, so much darker.
    let mut dark = Scene::new();
    dark.background = Color::from_hex(0x000000);
    dark.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        Material::Standard(StandardMaterial::new(Color::from_hex(0xffffff))),
    )));
    let unlit = center(&hr.render_to_rgba(&mut dark, &unit_camera()).unwrap());
    assert!(
        (unlit[0] as i32) < lit[0] as i32 - 20,
        "removing the lights changed nothing: lit {lit:?} vs unlit {unlit:?}"
    );
}

#[test]
fn the_depth_buffer_keeps_the_nearer_surface() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);

    let mut far = full_quad(Material::Basic(BasicMaterial::new(Color::from_hex(
        0x00ff00,
    ))));
    far.position = Vector3::new(0.0, 0.0, -0.5);
    let mut near = full_quad(Material::Basic(BasicMaterial::new(Color::from_hex(
        0xff0000,
    ))));
    near.position = Vector3::new(0.0, 0.0, 0.5);
    // Added far-first and near-second, then the reverse: the depth test, not
    // submission order, decides.
    scene.add(far);
    scene.add(near);

    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    assert!(
        close(center(&rgba), [255, 0, 0, 255], 2),
        "the far quad won: {:?}",
        center(&rgba)
    );
}

#[test]
fn transparent_surfaces_blend_over_what_is_behind_them() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);

    let mut back = full_quad(Material::Basic(BasicMaterial::new(Color::from_hex(
        0xff0000,
    ))));
    back.position = Vector3::new(0.0, 0.0, -0.5);
    scene.add(back);

    let mut glass = BasicMaterial::new(Color::from_hex(0x0000ff));
    glass.transparent = true;
    glass.opacity = 0.5;
    let mut front = full_quad(Material::Basic(glass));
    front.position = Vector3::new(0.0, 0.0, 0.5);
    scene.add(front);

    let px = center(&hr.render_to_rgba(&mut scene, &unit_camera()).unwrap());
    assert!(
        px[0] > 20 && px[2] > 20,
        "not a blend of both quads: {px:?}"
    );
    assert!(
        px[0] < 250 && px[2] < 250,
        "not a blend of both quads: {px:?}"
    );
}

#[test]
fn a_base_colour_map_is_sampled_the_right_way_up() {
    let Some(mut hr) = renderer(1) else { return };
    // Row 0 is the top of the image once `flip_y` has been applied, which is
    // three.js's convention and the wgpu backend's.
    let data = vec![
        255, 0, 0, 255, // top-left: red
        0, 255, 0, 255, // top-right: green
        0, 0, 255, 255, // bottom-left: blue
        255, 255, 255, 255, // bottom-right: white
    ];
    let mut texture = Texture::new(2, 2, TextureFormat::Rgba8UnormSrgb, data);
    texture.mag_filter = TextureFilter::Nearest;
    texture.min_filter = TextureFilter::Nearest;

    let mut material = BasicMaterial::new(Color::WHITE);
    material.map = Some(std::sync::Arc::new(texture));

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(full_quad(Material::Basic(material)));

    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    assert_eq!(hr.stats().texture_uploads, 1);
    assert!(
        close(pixel_at(&rgba, 0.25, 0.25), [255, 0, 0, 255], 2),
        "top-left: {:?}",
        pixel_at(&rgba, 0.25, 0.25)
    );
    assert!(
        close(pixel_at(&rgba, 0.75, 0.25), [0, 255, 0, 255], 2),
        "top-right: {:?}",
        pixel_at(&rgba, 0.75, 0.25)
    );
    assert!(
        close(pixel_at(&rgba, 0.25, 0.75), [0, 0, 255, 255], 2),
        "bottom-left: {:?}",
        pixel_at(&rgba, 0.25, 0.75)
    );
    assert!(
        close(pixel_at(&rgba, 0.75, 0.75), [255, 255, 255, 255], 2),
        "bottom-right: {:?}",
        pixel_at(&rgba, 0.75, 0.75)
    );

    // A second frame reuses the upload.
    let _ = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    assert_eq!(hr.stats().texture_uploads, 0);
    assert_eq!(hr.stats().geometry_uploads, 0);
}

#[test]
fn vertex_colours_modulate_the_material() {
    let Some(mut hr) = renderer(1) else { return };
    let mut geometry = PlaneGeometry::new(2.0, 2.0);
    let count = geometry.get_attribute("position").unwrap().count();
    geometry.set_attribute(
        "color",
        BufferAttribute::new(vec![0.0; count * 3], 3), // black everywhere
    );
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0xff0000);
    scene.add(Object3D::mesh(Mesh::new(
        geometry,
        Material::Basic(BasicMaterial::new(Color::WHITE)),
    )));

    let px = center(&hr.render_to_rgba(&mut scene, &unit_camera()).unwrap());
    assert!(
        close(px, [0, 0, 0, 255], 2),
        "a black vertex colour should black out a white material: {px:?}"
    );
}

#[test]
fn every_instance_of_an_instanced_mesh_is_drawn() {
    let Some(mut hr) = renderer(1) else { return };
    let mut instanced = InstancedMesh::new(
        PlaneGeometry::new(0.5, 0.5),
        Material::Basic(BasicMaterial::new(Color::from_hex(0xff0000))),
        2,
    );
    instanced.set_matrix_at(0, Matrix4::translation(Vector3::new(-0.5, 0.0, 0.0)));
    instanced.set_matrix_at(1, Matrix4::translation(Vector3::new(0.5, 0.0, 0.0)));

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::instanced_mesh(instanced));

    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    assert_eq!(
        hr.stats().draw_calls,
        1,
        "instances should be one draw call"
    );
    assert_eq!(hr.stats().triangles, 4, "two quads' worth of triangles");
    assert!(
        close(pixel_at(&rgba, 0.25, 0.5), [255, 0, 0, 255], 2),
        "left instance missing: {:?}",
        pixel_at(&rgba, 0.25, 0.5)
    );
    assert!(
        close(pixel_at(&rgba, 0.75, 0.5), [255, 0, 0, 255], 2),
        "right instance missing: {:?}",
        pixel_at(&rgba, 0.75, 0.5)
    );
    assert_eq!(
        pixel_at(&rgba, 0.5, 0.5),
        [0, 0, 0, 255],
        "the gap between instances should be background"
    );
}

#[test]
fn points_draw_as_sized_sprites() {
    let Some(mut hr) = renderer(1) else { return };
    let mut geometry = BufferGeometry::new();
    geometry.set_attribute("position", BufferAttribute::new(vec![0.0, 0.0, 0.0], 3));
    let mut material = PointsMaterial::new(Color::from_hex(0x00ff00), 16.0);
    material.size_attenuation = false;

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::points(Points::new(
        geometry,
        Material::Points(material),
    )));

    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    assert!(
        close(center(&rgba), [0, 255, 0, 255], 2),
        "no point at the centre: {:?}",
        center(&rgba)
    );
    assert_eq!(
        pixel_at(&rgba, 0.05, 0.05),
        [0, 0, 0, 255],
        "a 16 px point should not reach the corner"
    );
}

#[test]
fn line_segments_draw() {
    let Some(mut hr) = renderer(1) else { return };
    let mut geometry = BufferGeometry::new();
    geometry.set_attribute(
        "position",
        BufferAttribute::new(vec![-0.9, 0.0, 0.0, 0.9, 0.0, 0.0], 3),
    );
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::line_segments(LineSegments::new(
        geometry,
        Material::Line(LineBasicMaterial::new(Color::from_hex(0xffffff))),
    )));

    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    assert_eq!(hr.stats().draw_calls, 1);
    let drawn = rgba.chunks_exact(4).filter(|px| px[0] > 128).count();
    assert!(
        drawn >= SIZE as usize / 2,
        "the line barely drew: {drawn} px"
    );
    assert_eq!(hr.stats().triangles, 0, "a line list has no triangles");
}

#[test]
fn msaa_produces_partial_coverage_on_a_diagonal_edge() {
    // A diagonal edge is either in or out of a pixel without MSAA, and
    // partially in with it. Counting the in-between pixels is the difference.
    fn edge_pixels(msaa: u32) -> Option<usize> {
        let mut hr = renderer(msaa)?;
        let mut geometry = BufferGeometry::new();
        geometry.set_attribute(
            "position",
            BufferAttribute::new(vec![-0.9, -0.9, 0.0, 0.9, -0.9, 0.0, -0.9, 0.9, 0.0], 3),
        );
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        scene.add(Object3D::mesh(Mesh::new(
            geometry,
            Material::Basic(BasicMaterial::new(Color::WHITE)),
        )));
        let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
        Some(
            rgba.chunks_exact(4)
                .filter(|px| px[0] > 8 && px[0] < 247)
                .count(),
        )
    }

    let Some(aliased) = edge_pixels(1) else {
        return;
    };
    let Some(smoothed) = edge_pixels(4) else {
        return;
    };
    assert_eq!(aliased, 0, "1x sampling produced partial coverage");
    assert!(
        smoothed > 10,
        "4x MSAA produced only {smoothed} partially covered pixels"
    );
}

#[test]
fn caches_survive_between_frames_and_can_be_cleared() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = Scene::new();
    scene.add(full_quad(Material::Basic(BasicMaterial::new(Color::WHITE))));

    hr.render(&mut scene, &unit_camera()).unwrap();
    assert_eq!(hr.stats().geometry_uploads, 1);
    hr.render(&mut scene, &unit_camera()).unwrap();
    assert_eq!(hr.stats().geometry_uploads, 0, "geometry re-uploaded");

    assert_eq!(hr.renderer_mut().cache_sizes().0, 1);
    hr.renderer_mut().clear_caches();
    assert_eq!(hr.renderer_mut().cache_sizes(), (0, 0));
    hr.render(&mut scene, &unit_camera()).unwrap();
    assert_eq!(
        hr.stats().geometry_uploads,
        1,
        "cleared cache did not refill"
    );
}

#[test]
fn unused_cache_entries_are_evicted() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = Scene::new();
    scene.add(full_quad(Material::Basic(BasicMaterial::new(Color::WHITE))));
    hr.renderer_mut().set_cache_retention(3);

    hr.render(&mut scene, &unit_camera()).unwrap();
    assert_eq!(hr.renderer_mut().cache_sizes().0, 1);

    // The same scene keeps its entry alive however many frames pass.
    for _ in 0..8 {
        hr.render(&mut scene, &unit_camera()).unwrap();
    }
    assert_eq!(hr.renderer_mut().cache_sizes().0, 1);
    assert_eq!(hr.stats().geometry_uploads, 0);

    // An empty scene stops touching it, and it goes.
    let mut empty = Scene::new();
    for _ in 0..5 {
        hr.render(&mut empty, &unit_camera()).unwrap();
    }
    assert_eq!(
        hr.renderer_mut().cache_sizes().0,
        0,
        "a geometry nothing has drawn for 5 frames is still resident"
    );
}

#[test]
fn a_skipped_object_does_not_shift_another_objects_instance_buffer() {
    // The instance matrices live in a per-frame buffer list. An object dropped
    // between upload and encode used to shift every later instanced draw onto
    // the wrong buffer; this is the scene that would show it.
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);

    // An instanced mesh whose `position` is 2D: it survives collection, so it
    // gets a matrix buffer at upload, and then produces no vertices and draws
    // nothing. Nearest, so it sorts first.
    let mut flat = BufferGeometry::new();
    flat.set_attribute(
        "position",
        BufferAttribute::new(vec![0.0, 0.0, 1.0, 1.0], 2),
    );
    let mut ghost = InstancedMesh::new(flat, Material::Basic(BasicMaterial::new(Color::WHITE)), 3);
    ghost.set_matrix_at(0, Matrix4::translation(Vector3::new(0.0, 0.9, 0.0)));
    let mut ghost = Object3D::instanced_mesh(ghost);
    ghost.position = Vector3::new(0.0, 0.0, 0.9);
    scene.add(ghost);

    let mut instanced = InstancedMesh::new(
        PlaneGeometry::new(0.5, 0.5),
        Material::Basic(BasicMaterial::new(Color::from_hex(0x00ff00))),
        2,
    );
    instanced.set_matrix_at(0, Matrix4::translation(Vector3::new(-0.5, 0.0, 0.0)));
    instanced.set_matrix_at(1, Matrix4::translation(Vector3::new(0.5, 0.0, 0.0)));
    scene.add(Object3D::instanced_mesh(instanced));

    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    assert_eq!(hr.stats().skipped, 1);
    assert_eq!(hr.stats().draw_calls, 1);
    for fx in [0.25, 0.75] {
        assert!(
            close(pixel_at(&rgba, fx, 0.5), [0, 255, 0, 255], 2),
            "instance at x={fx} landed wrong: {:?}",
            pixel_at(&rgba, fx, 0.5)
        );
    }
}

#[test]
fn resizing_reallocates_the_target() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(full_quad(Material::Basic(BasicMaterial::new(Color::WHITE))));

    hr.resize(37, 21).unwrap();
    assert_eq!(hr.size(), (37, 21));
    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    assert_eq!(rgba.len(), 37 * 21 * 4);
    // 37 pixels is not a multiple of the 256-byte row alignment, so this also
    // checks the readback un-pads correctly.
    assert!(
        rgba.chunks_exact(4).all(|px| px[0] > 250),
        "un-padding the readback rows went wrong"
    );
}

#[test]
fn an_offscreen_target_can_be_driven_directly() {
    let Ok(device) = MetalDevice::new() else {
        return;
    };
    let mut renderer = threers::metal::MetalRenderer::with_device(device.clone()).unwrap();
    let target = MetalRenderTarget::new(
        &device,
        32,
        16,
        threers::metal::enums::pixel_format::RGBA8_UNORM_SRGB,
        1,
    )
    .unwrap();
    assert_eq!(target.size(), (32, 16));
    assert_eq!(target.sample_count(), 1);

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0000ff);
    let stats = renderer
        .render(&mut scene, &unit_camera(), &target.attachments())
        .unwrap();
    assert_eq!(stats.draw_calls, 0);

    let rgba = target.read_rgba(&device).unwrap();
    assert_eq!(rgba.len(), 32 * 16 * 4);
    assert!(close(
        [rgba[0], rgba[1], rgba[2], rgba[3]],
        [0, 0, 255, 255],
        2
    ));
}

#[test]
fn the_bgra_surface_format_draws_and_reads_back_in_rgba_order() {
    // `MetalSurface` presents BGRA8_UNORM_SRGB with MSAA, and a window cannot
    // be inspected from a test — so exercise that exact pipeline and readback
    // path against an offscreen target instead.
    let Ok(device) = MetalDevice::new() else {
        return;
    };
    let mut renderer = threers::metal::MetalRenderer::with_device(device.clone()).unwrap();
    let target = MetalRenderTarget::new(
        &device,
        32,
        32,
        threers::metal::enums::pixel_format::BGRA8_UNORM_SRGB,
        4,
    )
    .unwrap();
    assert_eq!(target.sample_count(), 4);

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(full_quad(Material::Basic(BasicMaterial::new(
        Color::from_hex(0xff0000),
    ))));
    renderer
        .render(&mut scene, &unit_camera(), &target.attachments())
        .unwrap();

    let rgba = target.read_rgba(&device).unwrap();
    assert!(
        close([rgba[0], rgba[1], rgba[2], rgba[3]], [255, 0, 0, 255], 2),
        "BGRA readback is not in RGBA order: {:?}",
        &rgba[0..4]
    );
}

/// The stereo path visionOS runs on: two eyes, one pass, one texture array —
/// the compositor's `layered` layout, reproduced offscreen.
#[test]
fn both_eyes_draw_into_their_own_slice_in_one_pass() {
    let Ok(device) = MetalDevice::new() else {
        return;
    };
    let mut renderer = threers::metal::MetalRenderer::with_device(device.clone()).unwrap();
    let target = MetalRenderTarget::layered(
        &device,
        64,
        64,
        threers::metal::enums::pixel_format::RGBA8_UNORM_SRGB,
        2,
    )
    .unwrap();
    assert_eq!(target.slices(), 2);

    // A narrow post at the origin, and two eyes 2 units apart looking straight
    // ahead. Each eye sees the post on its own side of the frame.
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(0.2, 2.0),
        Material::Basic(BasicMaterial::new(Color::WHITE)),
    )));

    let eye = |x: f32, slice: u32| {
        let mut camera = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
        camera.position = Vector3::new(x, 0.0, 2.0);
        camera.target = Vector3::new(x, 0.0, 0.0);
        RenderView::from_camera(&camera).with_slice(slice)
    };
    let views = [eye(-0.5, 0), eye(0.5, 1)];

    let stats = renderer
        .render_views(&mut scene, &views, &target.attachments())
        .unwrap();
    assert_eq!(
        stats.draw_calls, 1,
        "a layered pass should encode one draw for both eyes"
    );

    // The left eye sits left of the post, so it sees it to the right; the right
    // eye, the other way about.
    let left = target.read_rgba_slice(&device, 0).unwrap();
    let right = target.read_rgba_slice(&device, 1).unwrap();
    let at = |px: &[u8], fx: f32| {
        let x = ((fx * 64.0) as usize).min(63);
        px[(32 * 64 + x) * 4]
    };
    assert!(at(&left, 0.75) > 200, "left eye: post not on the right");
    assert!(at(&left, 0.25) < 40, "left eye: something on the left");
    assert!(at(&right, 0.25) > 200, "right eye: post not on the left");
    assert!(at(&right, 0.75) < 40, "right eye: something on the right");
    assert_ne!(left, right, "both slices came out identical");
}

/// Eyes that cannot share a pass — one texture each — still both draw.
#[test]
fn views_with_separate_targets_render_one_pass_each() {
    let Ok(device) = MetalDevice::new() else {
        return;
    };
    let mut renderer = threers::metal::MetalRenderer::with_device(device.clone()).unwrap();
    let format = threers::metal::enums::pixel_format::RGBA8_UNORM_SRGB;
    let left = MetalRenderTarget::new(&device, 32, 32, format, 1).unwrap();
    let right = MetalRenderTarget::new(&device, 32, 32, format, 1).unwrap();

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(full_quad(Material::Basic(BasicMaterial::new(Color::WHITE))));

    let camera = unit_camera();
    let views = unsafe {
        [
            RenderView::from_camera(&camera)
                .with_textures(left.color_texture(), left.depth_texture()),
            RenderView::from_camera(&camera)
                .with_textures(right.color_texture(), right.depth_texture()),
        ]
    };
    let stats = renderer
        .render_views(&mut scene, &views, &left.attachments())
        .unwrap();
    assert_eq!(stats.draw_calls, 2, "one draw per view when not layered");

    for (name, target) in [("left", &left), ("right", &right)] {
        let rgba = target.read_rgba(&device).unwrap();
        assert!(
            rgba[0] > 250,
            "{name} target did not receive its pass: {:?}",
            &rgba[0..4]
        );
    }
}

/// visionOS accepts reverse-Z depth and nothing else, so the depth test has to
/// invert with it: near at 1, far at 0, cleared to 0, compared with `greater`.
#[test]
fn reverse_z_keeps_the_nearer_surface() {
    let Ok(device) = MetalDevice::new() else {
        return;
    };
    let mut renderer = threers::metal::MetalRenderer::with_device(device.clone()).unwrap();
    let target = MetalRenderTarget::new(
        &device,
        32,
        32,
        threers::metal::enums::pixel_format::RGBA8_UNORM_SRGB,
        1,
    )
    .unwrap();

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let mut far = full_quad(Material::Basic(BasicMaterial::new(Color::from_hex(
        0x00ff00,
    ))));
    far.position = Vector3::new(0.0, 0.0, -0.5);
    scene.add(far);
    let mut near = full_quad(Material::Basic(BasicMaterial::new(Color::from_hex(
        0xff0000,
    ))));
    near.position = Vector3::new(0.0, 0.0, 0.5);
    scene.add(near);

    // Ordinary projection, then z flipped: what maps to 0 now maps to 1.
    let camera = unit_camera();
    let mut projection = camera.projection_matrix();
    for row in [2usize, 6, 10, 14] {
        projection.elements[row] = -projection.elements[row];
    }
    projection.elements[10] += projection.elements[11];
    projection.elements[14] += projection.elements[15];

    let view = RenderView {
        projection_matrix: projection,
        ..RenderView::from_camera(&camera)
    };
    let pass = target.attachments().with_reverse_z(true);
    assert!(pass.reverse_z());
    renderer.render_views(&mut scene, &[view], &pass).unwrap();

    let rgba = target.read_rgba(&device).unwrap();
    assert!(
        close([rgba[0], rgba[1], rgba[2], rgba[3]], [255, 0, 0, 255], 2),
        "reverse-Z kept the far quad: {:?}",
        &rgba[0..4]
    );
}

#[test]
fn geometry_without_positions_is_counted_not_drawn() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = Scene::new();
    scene.add(Object3D::mesh(Mesh::new(
        BufferGeometry::new(),
        Material::Basic(BasicMaterial::new(Color::WHITE)),
    )));
    let stats = hr.render(&mut scene, &unit_camera()).unwrap();
    assert_eq!(stats.draw_calls, 0);
    assert_eq!(stats.skipped, 1);
}

#[test]
fn camera_layers_filter_what_is_drawn() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let mut quad = full_quad(Material::Basic(BasicMaterial::new(Color::WHITE)));
    quad.layers.set(2);
    scene.add(quad);

    let camera = unit_camera();
    assert!(camera.layers().test(&threers::core::Layers::default()));
    let rgba = hr.render_to_rgba(&mut scene, &camera).unwrap();
    assert_eq!(
        center(&rgba),
        [0, 0, 0, 255],
        "a mesh on layer 2 should be invisible to a default-layer camera"
    );
}

#[test]
fn headless_render_scale_upscales_on_readback() {
    let Some(mut hr) = (match MetalHeadlessRenderer::builder()
            .size(SIZE, SIZE)
            .render_scale(2)
            .build()
        {
            Ok(r) => Some(r),
            Err(MetalError::NoDevice) => {
                eprintln!("skipping: no Metal device on this machine");
                None
            }
            Err(e) => panic!("{e}"),
        }) else {
        return;
    };
    assert_eq!(hr.export_size(), (SIZE, SIZE));
    assert_eq!(hr.render_size(), (SIZE / 2, SIZE / 2));
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(full_quad(Material::Basic(BasicMaterial::new(Color::WHITE))));
    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    assert_eq!(rgba.len(), (SIZE * SIZE * 4) as usize);
    // Catmull-Rom must cover the full source, not a corner crop.
    assert!(
        center(&rgba)[0] > 200,
        "upscaled white quad should stay bright at center: {:?}",
        center(&rgba)
    );
}

#[test]
fn headless_supersample_downsample() {
    let Some(mut hr) = (match MetalHeadlessRenderer::builder()
            .size(SIZE, SIZE)
            .supersample(4)
            .build()
        {
            Ok(r) => Some(r),
            Err(MetalError::NoDevice) => {
                eprintln!("skipping: no Metal device on this machine");
                None
            }
            Err(e) => panic!("{e}"),
        }) else {
        return;
    };
    assert_eq!(hr.export_size(), (SIZE, SIZE));
    assert_eq!(hr.render_size(), (SIZE * 4, SIZE * 4));
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(full_quad(Material::Basic(BasicMaterial::new(Color::WHITE))));
    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    assert_eq!(rgba.len(), (SIZE * SIZE * 4) as usize);
}

#[test]
fn headless_render_without_bloom() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(full_quad(Material::Basic(BasicMaterial::new(Color::WHITE))));
    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    assert_eq!(rgba.len(), (SIZE * SIZE * 4) as usize);
}

#[test]
fn postfx_pipelines_compile() {
    let device = match MetalDevice::new() {
        Ok(d) => d,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    threers::metal::PostFxChain::new(&device).expect("postfx pipelines should compile");
}

#[test]
fn headless_bloom_does_not_panic() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(full_quad(Material::Basic(BasicMaterial::new(Color::WHITE))));
    hr.set_bloom(0.5, 0.2, 2.0);
    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    assert_eq!(rgba.len(), (SIZE * SIZE * 4) as usize);
    assert!(center(&rgba)[0] > 200, "white quad should survive bloom composite");
}

// ------------------------------------------------------------- wireframe

// The probes `tests/wgpu_wireframe.rs` uses, so both backends are held to the
// same picture. They assume the `SIZE` image seen through `unit_camera`.
#[path = "wire_probe/mod.rs"]
mod wire_probe;
use wire_probe::{assert_outline_only, diagonals_drawn, lit_at, wire_material};
const _: () = assert!(SIZE == wire_probe::SIZE);

fn wire_quad(polygons: bool) -> BufferGeometry {
    let mut g = wire_probe::plane();
    // PlaneGeometry records its quad itself; the control drops it to get
    // three.js's per-triangle wireframe.
    g.polygons = None;
    if polygons {
        g.set_polygons(vec![2]);
        assert!(g.polygon_counts().is_some(), "plane should accept one quad");
    }
    g
}

fn wire_scene(polygons: bool) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::mesh(Mesh::new(wire_quad(polygons), wire_material())));
    scene
}

#[test]
fn wireframe_quad_draws_its_outline_without_the_diagonal() {
    let Some(mut hr) = renderer(1) else { return };

    // Control: without a polygon table it is three.js's per-triangle
    // wireframe, so the plane's own diagonal is drawn. Proves the probe can
    // see a diagonal at all.
    let rgba = hr.render_to_rgba(&mut wire_scene(false), &unit_camera()).unwrap();
    assert_eq!(
        diagonals_drawn(&rgba),
        1,
        "triangle wireframe should show exactly one diagonal"
    );

    let rgba = hr.render_to_rgba(&mut wire_scene(true), &unit_camera()).unwrap();
    assert_outline_only(&rgba, "metal");
    assert_eq!(hr.stats().triangles, 2, "a wireframe still reports its triangles");
}

#[test]
fn instanced_wireframe_quads_draw_outlines_only() {
    let Some(mut hr) = renderer(1) else { return };
    // Two instances: one where the single-mesh test draws its quad, one
    // shifted far enough to sit beside it rather than over it.
    let mut instanced = InstancedMesh::new(wire_quad(true), wire_material(), 2);
    instanced.set_matrix_at(0, Matrix4::identity());
    instanced.set_matrix_at(1, Matrix4::translation(Vector3::new(5.0, 0.0, 0.0)));
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::instanced_mesh(instanced));
    let rgba = hr.render_to_rgba(&mut scene, &unit_camera()).unwrap();
    assert_outline_only(&rgba, "metal instanced");
}

/// The wgpu backend draws the same wireframe (1px hardware lines from the
/// same line list), so the two should light the same pixels.
#[test]
fn wireframe_quad_matches_the_wgpu_backend() {
    let Some(mut hr) = renderer(1) else { return };
    let metal = hr.render_to_rgba(&mut wire_scene(true), &unit_camera()).unwrap();

    let Ok(mut wr) = threers::HeadlessRenderer::builder()
        .size(SIZE, SIZE)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
    else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    wr.set_msaa(1);
    let wgpu = wr.render_to_rgba(&mut wire_scene(true), &unit_camera());
    assert_outline_only(&wgpu, "wgpu");

    let mut lit = 0;
    let mut differ = Vec::new();
    for y in 0..SIZE as i32 {
        for x in 0..SIZE as i32 {
            let (m, w) = (lit_at(&metal, x, y), lit_at(&wgpu, x, y));
            lit += m as usize;
            if m != w {
                differ.push((x, y, m));
            }
        }
    }
    eprintln!("wireframe: {lit} lit pixels, {} differ from wgpu", differ.len());
    assert!(lit > 100, "metal lit only {lit} pixels");
    assert!(
        differ.len() * 50 <= lit,
        "metal and wgpu disagree on {} of {lit} lit pixels: {:?}",
        differ.len(),
        &differ[..differ.len().min(16)]
    );
}

// ------------------------------------------------- drawn where it is picked
//
// A skinned mesh and a sprite are both positioned by more than their world
// matrix — bones, a billboard — so each backend and the raycaster used to
// put them in a different place. These hold all three to one picture: the two
// backends must cover the same pixels, and a ray through a pixel the mesh
// covers solidly must hit it, one through a pixel it clearly misses must not.

/// wgpu at `SIZE`, one sample per pixel, RGBA, or `None` without an adapter.
fn wgpu_renderer() -> Option<threers::HeadlessRenderer> {
    let mut r = threers::HeadlessRenderer::builder()
        .size(SIZE, SIZE)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()?;
    r.set_msaa(1);
    Some(r)
}

fn lit(rgba: &[u8], x: i32, y: i32) -> bool {
    wire_probe::lit_at(rgba, x, y)
}

/// Pixels on which the two images disagree about coverage, and how many each
/// lights.
fn coverage_diff(a: &[u8], b: &[u8]) -> (usize, usize, usize) {
    let (mut differ, mut lit_a, mut lit_b) = (0, 0, 0);
    for y in 0..SIZE as i32 {
        for x in 0..SIZE as i32 {
            let (p, q) = (lit(a, x, y), lit(b, x, y));
            lit_a += p as usize;
            lit_b += q as usize;
            differ += (p != q) as usize;
        }
    }
    (differ, lit_a, lit_b)
}

/// Cast a ray through the centre of every pixel whose 3×3 neighbourhood is
/// uniformly covered or uniformly empty in `rgba`, and return
/// `(interior pixels, exterior pixels, mismatches)`. Edge pixels, where a
/// rasteriser and a ray test may round either way, are left out.
fn picks_match_pixels(
    rgba: &[u8],
    scene: &mut Scene,
    camera: &dyn Camera,
    target: threers::core::ObjectId,
) -> (usize, usize, Vec<(i32, i32, bool)>) {
    use threers::core::Raycaster;
    use threers::math::Vector2;
    scene
        .arena
        .update_world_matrices(scene.root, Matrix4::identity());
    let (mut inside, mut outside, mut wrong) = (0, 0, Vec::new());
    for y in 1..SIZE as i32 - 1 {
        for x in 1..SIZE as i32 - 1 {
            let here = lit(rgba, x, y);
            let uniform = (-1..=1).all(|dy| (-1..=1).all(|dx| lit(rgba, x + dx, y + dy) == here));
            if !uniform {
                continue;
            }
            let ndc = Vector2::new(
                (x as f32 + 0.5) / SIZE as f32 * 2.0 - 1.0,
                1.0 - (y as f32 + 0.5) / SIZE as f32 * 2.0,
            );
            let mut rc = Raycaster::default();
            rc.set_from_camera(ndc, camera);
            let hit = rc
                .intersect_object(&scene.arena, scene.root, true)
                .iter()
                .any(|h| h.object == target);
            if here {
                inside += 1;
            } else {
                outside += 1;
            }
            if hit != here {
                wrong.push((x, y, here));
            }
        }
    }
    (inside, outside, wrong)
}

/// A skinned strip: a non-identity mesh matrix, two posed bones, and weights
/// that do not sum to one. three.js draws that at
/// `Σ wᵢ·Bᵢ·p + (1 − Σ wᵢ)·translation(matrix_world)`; both backends, the
/// raycaster and the path tracer must agree on it.
fn skinned_scene() -> (Scene, threers::core::ObjectId) {
    use threers::core::{Bone, ObjectId, Skeleton, SkinnedMesh};
    use threers::math::Euler;
    let mut geom = PlaneGeometry::with_segments(0.8, 1.2, 2, 6);
    let count = geom.get_attribute("position").unwrap().count();
    let ys: Vec<f32> = (0..count)
        .map(|i| geom.get_attribute("position").unwrap().array[i * 3 + 1])
        .collect();
    let mut joints = vec![0.0f32; count * 4];
    let mut weights = vec![0.0f32; count * 4];
    for i in 0..count {
        // Lower part on bone 0, upper on bone 1, the middle row on both —
        // and every vertex's weights sum to 0.85, not 1.
        let y = ys[i];
        let (w0, w1) = if y < -0.05 {
            (0.85, 0.0)
        } else if y > 0.05 {
            (0.0, 0.85)
        } else {
            (0.5, 0.35)
        };
        joints[i * 4] = 0.0;
        joints[i * 4 + 1] = 1.0;
        weights[i * 4] = w0;
        weights[i * 4 + 1] = w1;
    }
    geom.set_attribute("joint", BufferAttribute::new(joints, 4));
    geom.set_attribute("weight", BufferAttribute::new(weights, 4));
    let mut skeleton = Skeleton::new(vec![
        Bone {
            node: ObjectId::default(),
            inverse_bind: Matrix4::identity(),
        },
        Bone {
            node: ObjectId::default(),
            inverse_bind: Matrix4::identity(),
        },
    ]);
    let rot = |a: f32| Matrix4::from_quaternion(Euler::new(0.0, 0.0, a).to_quaternion());
    skeleton.bone_matrices = vec![
        Matrix4::translation(Vector3::new(-0.15, -0.1, 0.0)).multiply(&rot(0.2)),
        Matrix4::translation(Vector3::new(0.1, 0.05, 0.0)).multiply(&rot(-0.5)),
    ];
    let mut obj = Object3D::skinned_mesh(SkinnedMesh::new(
        geom,
        BasicMaterial::new(Color::WHITE).into(),
        skeleton,
    ));
    // Translated, turned about the view axis, scaled: the translation shows
    // through the missing 0.15 of weight, the rest cancels.
    obj.position = Vector3::new(0.4, -0.3, 0.0);
    obj.quaternion = Euler::new(0.0, 0.0, 0.7).to_quaternion();
    obj.scale = Vector3::new(1.3, 0.9, 1.0);
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let id = scene.add(obj);
    (scene, id)
}

#[test]
fn skinned_mesh_is_drawn_and_picked_in_one_place() {
    let Some(mut m) = renderer(1) else { return };
    let Some(mut w) = wgpu_renderer() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let cam = unit_camera();
    let (mut scene, id) = skinned_scene();
    let metal = m.render_to_rgba(&mut scene, &cam).unwrap();
    let wgpu = w.render_to_rgba(&mut scene, &cam);
    let (differ, lit_m, lit_w) = coverage_diff(&metal, &wgpu);
    eprintln!("skinned: metal lights {lit_m}, wgpu {lit_w}, {differ} differ");
    assert!(lit_m > 150, "metal drew only {lit_m} pixels");
    assert!(
        differ * 40 <= lit_m,
        "metal and wgpu disagree on {differ} of {lit_m} pixels"
    );

    // three.js's formula, by hand, for every vertex.
    let world = scene.arena.get(id).unwrap().matrix_world;
    let threers::core::ObjectKind::SkinnedMesh(sm) = &scene.arena.get(id).unwrap().kind else {
        unreachable!()
    };
    let local = sm.posed_positions(&world).unwrap();
    let (pos, j, wt) = (
        sm.geometry.get_attribute("position").unwrap(),
        sm.geometry.get_attribute("joint").unwrap(),
        sm.geometry.get_attribute("weight").unwrap(),
    );
    let t = Vector3::new(world.elements[12], world.elements[13], world.elements[14]);
    for v in 0..pos.count() {
        let p = Vector3::new(pos.array[v * 3], pos.array[v * 3 + 1], pos.array[v * 3 + 2]);
        let mut expect = Vector3::ZERO;
        let mut total = 0.0;
        for k in 0..4 {
            let b = sm.skeleton.bone_matrices[j.array[v * 4 + k] as usize];
            expect = expect + p.apply_matrix4(&b) * wt.array[v * 4 + k];
            total += wt.array[v * 4 + k];
        }
        assert!((total - 0.85f32).abs() < 1e-6);
        expect = expect + t * (1.0 - total);
        let got =
            Vector3::new(local[v * 3], local[v * 3 + 1], local[v * 3 + 2]).apply_matrix4(&world);
        assert!(
            (got - expect).length() < 1e-5,
            "vertex {v}: {got:?} vs three.js {expect:?}"
        );
    }

    for (name, img) in [("metal", &metal), ("wgpu", &wgpu)] {
        let (inside, outside, wrong) = picks_match_pixels(img, &mut scene, &cam, id);
        eprintln!(
            "skinned picks vs {name}: {inside} inside, {outside} outside, {} wrong",
            wrong.len()
        );
        assert!(inside > 80 && outside > 1000);
        assert!(
            wrong.is_empty(),
            "{name}: picks disagree with the pixels at {wrong:?}"
        );
    }
}

/// A sprite anchored off-centre and turned, under an orthographic camera and
/// under a perspective one with size attenuation off.
#[test]
fn sprite_is_drawn_and_picked_in_one_place() {
    use threers::cameras::PerspectiveCamera;
    use threers::core::Sprite;
    use threers::materials::SpriteMaterial;
    let Some(mut m) = renderer(1) else { return };
    let Some(mut w) = wgpu_renderer() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let sprite_scene = |size_attenuation: bool, scale: f32| {
        let mut s = Sprite::new(Material::Sprite(SpriteMaterial {
            rotation: 0.6,
            size_attenuation,
            ..SpriteMaterial::default()
        }));
        s.center = threers::math::Vector2::new(0.15, 0.8);
        let mut obj = Object3D::sprite(s);
        obj.position = Vector3::new(0.1, -0.2, 0.0);
        obj.scale = Vector3::new(1.0 * scale, 0.5 * scale, 1.0);
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        let id = scene.add(obj);
        (scene, id)
    };
    let mut persp = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
    persp.position = Vector3::new(0.3, 0.2, 3.0);
    persp.look_at(Vector3::ZERO);
    let ortho = unit_camera();
    let cases: [(&str, &dyn Camera, bool, f32); 3] = [
        ("ortho", &ortho, true, 1.0),
        ("perspective", &persp, true, 1.0),
        // Not attenuated: scaled by the view depth (~3), so a third the size.
        ("perspective, fixed size", &persp, false, 0.33),
    ];
    for (name, cam, atten, scale) in cases {
        let (mut scene, id) = sprite_scene(atten, scale);
        let metal = m.render_to_rgba(&mut scene, cam).unwrap();
        let wgpu = w.render_to_rgba(&mut scene, cam);
        let (differ, lit_m, lit_w) = coverage_diff(&metal, &wgpu);
        eprintln!("sprite {name}: metal lights {lit_m}, wgpu {lit_w}, {differ} differ");
        assert!(lit_m > 150, "{name}: metal drew only {lit_m} pixels");
        assert!(
            differ * 40 <= lit_m,
            "{name}: backends disagree on {differ} of {lit_m}"
        );
        for (backend, img) in [("metal", &metal), ("wgpu", &wgpu)] {
            let (inside, outside, wrong) = picks_match_pixels(img, &mut scene, cam, id);
            eprintln!(
                "sprite {name} picks vs {backend}: {inside} in, {outside} out, {} wrong",
                wrong.len()
            );
            assert!(inside > 80 && outside > 1000);
            assert!(
                wrong.is_empty(),
                "{name}/{backend}: picks disagree at {wrong:?}"
            );
        }
    }
}

// ------------------------------------------------- groups, drawRange, morphs

/// The colour class of a pixel: 0 background, 1 red, 2 green, 3 other.
fn colour_class(rgba: &[u8], x: i32, y: i32) -> u8 {
    let i = (y as usize * SIZE as usize + x as usize) * 4;
    let (r, g, b) = (rgba[i], rgba[i + 1], rgba[i + 2]);
    match (r > 128, g > 128, b > 128) {
        (false, false, false) => 0,
        (true, false, false) => 1,
        (false, true, false) => 2,
        _ => 3,
    }
}

/// A pixel drawn in the wrong class: `(x, y, predicted, drawn)`.
type Mismatch = (i32, i32, u8, u8);

/// For every pixel whose 3×3 neighbourhood is one colour class, the class the
/// raycaster predicts — background on a miss, else the class of the material
/// at the nearest hit's `face.material_index` — against the class drawn.
/// Returns `(checked, per-class counts, mismatches)`.
fn classes_match_picks(
    rgba: &[u8],
    scene: &mut Scene,
    camera: &dyn Camera,
    class_of_material: &[u8],
) -> (usize, [usize; 4], Vec<Mismatch>) {
    use threers::core::Raycaster;
    use threers::math::Vector2;
    scene
        .arena
        .update_world_matrices(scene.root, Matrix4::identity());
    let (mut checked, mut counts, mut wrong) = (0, [0usize; 4], Vec::new());
    for y in 1..SIZE as i32 - 1 {
        for x in 1..SIZE as i32 - 1 {
            let here = colour_class(rgba, x, y);
            let uniform = (-1..=1)
                .all(|dy| (-1..=1).all(|dx| colour_class(rgba, x + dx, y + dy) == here));
            if !uniform {
                continue;
            }
            let ndc = Vector2::new(
                (x as f32 + 0.5) / SIZE as f32 * 2.0 - 1.0,
                1.0 - (y as f32 + 0.5) / SIZE as f32 * 2.0,
            );
            let mut rc = Raycaster::default();
            rc.set_from_camera(ndc, camera);
            let hits = rc.intersect_object(&scene.arena, scene.root, true);
            let predicted = match hits.first() {
                None => 0,
                Some(h) => class_of_material[h.face.unwrap().material_index],
            };
            checked += 1;
            counts[here as usize] += 1;
            if predicted != here {
                wrong.push((x, y, here, predicted));
            }
        }
    }
    (checked, counts, wrong)
}

fn flat(hex: u32) -> std::sync::Arc<Material> {
    std::sync::Arc::new(BasicMaterial::new(Color::from_hex(hex)).into())
}

/// three.js's `new Mesh(geometry, [green, red])` over a plane split into
/// groups: each group draws in its own material, the raycaster reports the
/// same `face.materialIndex` the pixel shows, and a draw range cuts both.
#[test]
fn groups_draw_their_materials_where_the_raycaster_reports_them() {
    let Some(mut hr) = renderer(1) else { return };
    let cam = unit_camera();
    let build = |range: Option<(u32, u32)>| {
        let mut g = PlaneGeometry::with_segments(1.6, 1.6, 4, 1);
        let n = g.index.as_ref().unwrap().len() as u32;
        assert_eq!(n, 24);
        g.add_group(0, 9, 1).add_group(9, 15, 0);
        if let Some((s, c)) = range {
            g.set_draw_range(s, c);
        }
        let mesh = Mesh::with_materials(std::sync::Arc::new(g), vec![flat(0x00ff00), flat(0xff0000)]);
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        scene.add(Object3D::mesh(mesh));
        scene
    };
    // Material 0 is green (class 2), material 1 red (class 1).
    let classes = [2u8, 1u8];
    for (name, range) in [("groups", None), ("groups + drawRange", Some((6, 12)))] {
        let mut scene = build(range);
        let rgba = hr.render_to_rgba(&mut scene, &cam).unwrap();
        let (checked, counts, wrong) = classes_match_picks(&rgba, &mut scene, &cam, &classes);
        eprintln!("{name}: {checked} checked, classes {counts:?}, {} wrong", wrong.len());
        assert!(counts[1] > 100 && counts[2] > 100, "{name}: both materials drawn: {counts:?}");
        assert!(counts[0] > 100, "{name}: background around the plane");
        assert_eq!(counts[3], 0, "{name}: no pixel in neither material");
        assert!(wrong.is_empty(), "{name}: pixels disagree with picks: {:?}", &wrong[..wrong.len().min(10)]);
    }
    // The cut range leaves less of each drawn.
    let full = hr.render_to_rgba(&mut build(None), &cam).unwrap();
    let cut = hr.render_to_rgba(&mut build(Some((6, 12))), &cam).unwrap();
    let lit_count = |img: &[u8]| (0..SIZE as i32 * SIZE as i32).filter(|i| lit(img, i % SIZE as i32, i / SIZE as i32)).count();
    assert!(lit_count(&cut) * 3 < lit_count(&full) * 2, "drawRange cut the plane");
}

/// A plane under morph targets — absolute, and relative ones describing the
/// same shapes — is drawn where the raycaster (three.js's `getVertexPosition`)
/// hits it, and not where the rest pose was.
#[test]
fn morphed_meshes_are_drawn_where_they_are_picked() {
    let Some(mut hr) = renderer(1) else { return };
    let cam = unit_camera();
    for relative in [false, true] {
        let mut g = PlaneGeometry::new(1.0, 1.0);
        let base = g.get_attribute("position").unwrap().array.clone();
        let (mut t0, mut t1) = (Vec::new(), Vec::new());
        for p in base.chunks_exact(3) {
            // Target 0 widens x by 1.6, target 1 shifts up by 0.4.
            if relative {
                t0.extend_from_slice(&[p[0] * 0.6, 0.0, 0.0]);
                t1.extend_from_slice(&[0.0, 0.4, 0.0]);
            } else {
                t0.extend_from_slice(&[p[0] * 1.6, p[1], p[2]]);
                t1.extend_from_slice(&[p[0], p[1] + 0.4, p[2]]);
            }
        }
        g.morph_targets_relative = relative;
        g.set_morph_attribute("position", vec![BufferAttribute::new(t0, 3), BufferAttribute::new(t1, 3)]);
        let mut mesh = Mesh::new(g, BasicMaterial::new(Color::WHITE).into());
        mesh.morph_influences = vec![1.0, 0.75];
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        let id = scene.add(Object3D::mesh(mesh));
        let rgba = hr.render_to_rgba(&mut scene, &cam).unwrap();
        let (inside, outside, wrong) = picks_match_pixels(&rgba, &mut scene, &cam, id);
        eprintln!("morph relative={relative}: {inside} in, {outside} out, {} wrong", wrong.len());
        assert!(wrong.is_empty(), "relative={relative}: {wrong:?}");
        // 1.6 × 1 at y ∈ [-0.2, 0.8]: y = 0.61 is covered and y = -0.42 is
        // not — the rest pose, y ∈ [-0.5, 0.5], was the reverse.
        assert!(lit(&rgba, 32, 12) && !lit(&rgba, 32, 45), "relative={relative}");
        assert!(lit(&rgba, 7, 25) && lit(&rgba, 56, 25), "widened to x = ±0.8");
        assert!(inside > 1000, "{inside}");
    }
}

/// three.js's chunk order on a skinned mesh with morph targets: the morph
/// moves the rest pose, then the bones carry it. The raycaster composes the
/// same way, and every pixel agrees with it.
#[test]
fn morph_targets_apply_before_skinning() {
    use threers::core::ObjectKind;
    let Some(mut m) = renderer(1) else { return };
    let cam = unit_camera();
    let (mut scene, id) = skinned_scene();
    let rest = m.render_to_rgba(&mut scene, &cam).unwrap();
    if let ObjectKind::SkinnedMesh(sm) = &mut scene.arena.get_mut(id).unwrap().kind {
        let mut g = (*sm.geometry).clone();
        let n = g.get_attribute("position").unwrap().count();
        let pos = g.get_attribute("position").unwrap().array.clone();
        // Relative: stretch the strip along its length.
        let delta: Vec<f32> = (0..n).flat_map(|v| [0.0, pos[v * 3 + 1] * 0.5, 0.0]).collect();
        g.morph_targets_relative = true;
        g.set_morph_attribute("position", vec![BufferAttribute::new(delta, 3)]);
        sm.geometry = std::sync::Arc::new(g);
        sm.update_morph_targets();
        sm.morph_influences = vec![0.8];
    }
    let morphed = m.render_to_rgba(&mut scene, &cam).unwrap();
    let (differ, lit_rest, lit_morph) = coverage_diff(&rest, &morphed);
    eprintln!("skinned morph: rest {lit_rest}, morphed {lit_morph}, {differ} differ");
    assert!(differ > 50, "the morph moved the skinned strip");
    let (inside, outside, wrong) = picks_match_pixels(&morphed, &mut scene, &cam, id);
    eprintln!("skinned morph picks: {inside} in, {outside} out, {} wrong", wrong.len());
    assert!(inside > 80 && outside > 1000);
    assert!(wrong.is_empty(), "picks disagree with the pixels at {wrong:?}");
}

/// Morph normals reach the lighting: a plane facing the camera whose normal
/// target turns it away from the light darkens, by the amount the CPU pose
/// (`posed_geometry`'s blended normal) predicts.
#[test]
fn morph_normals_change_the_shading() {
    let Some(mut hr) = renderer(1) else { return };
    let cam = unit_camera();
    let shade = |hr: &mut MetalHeadlessRenderer, influence: f32| {
        let mut g = PlaneGeometry::new(2.0, 2.0);
        let n = g.get_attribute("position").unwrap().count();
        // Absolute normal target: tilted 60° about x.
        let (s, c) = (60f32.to_radians().sin(), 60f32.to_radians().cos());
        let tn: Vec<f32> = (0..n).flat_map(|_| [0.0, -s, c]).collect();
        let tp = g.get_attribute("position").unwrap().array.clone();
        g.set_morph_attribute("position", vec![BufferAttribute::new(tp, 3)]);
        g.morph_attributes.insert("normal".into(), vec![BufferAttribute::new(tn, 3)]);
        g.bump_version();
        let mut mesh = Mesh::new(g, threers::materials::LambertMaterial::new(Color::WHITE).into());
        mesh.morph_influences = vec![influence];
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        scene.add(Object3D::mesh(mesh));
        let mut light = Object3D::light(DirectionalLight::new(Color::WHITE, 1.0));
        light.position = Vector3::new(0.0, 0.0, 1.0);
        scene.add(light);
        let rgba = hr.render_to_rgba(&mut scene, &cam).unwrap();
        center(&rgba)[0]
    };
    let flat_lit = shade(&mut hr, 0.0);
    let tilted = shade(&mut hr, 1.0);
    eprintln!("lambert centre: rest {flat_lit}, normal-morphed {tilted}");
    assert!(flat_lit > 100, "{flat_lit}");
    // cos 60° = 0.5 of the light in linear terms: clearly darker, not black.
    assert!(tilted + 30 < flat_lit && tilted > 40, "{tilted} vs {flat_lit}");
}

/// Per-pixel colour parity of two images: how many pixels differ in any
/// channel by more than `tol`, and how many are lit.
fn colour_diff(a: &[u8], b: &[u8], tol: i32) -> (usize, usize) {
    let (mut differ, mut lit_px) = (0, 0);
    for i in (0..a.len()).step_by(4) {
        lit_px += (a[i..i + 3].iter().any(|&v| v > 8)) as usize;
        differ += (0..3).any(|k| (a[i + k] as i32 - b[i + k] as i32).abs() > tol) as usize;
    }
    (differ, lit_px)
}

/// The same multi-material mesh — groups and a draw range — on both
/// backends: every pixel within a few levels.
#[test]
fn groups_and_draw_range_match_the_wgpu_backend() {
    let Some(mut m) = renderer(1) else { return };
    let Some(mut w) = wgpu_renderer() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let cam = unit_camera();
    let mut g = PlaneGeometry::with_segments(1.6, 1.6, 4, 1);
    g.add_group(0, 6, 2).add_group(6, 6, 0).add_group(12, 12, 1);
    g.set_draw_range(0, 18);
    // Opaque throughout: this is about groups, not blending. Per-group
    // transparency is held to three.js in `tests/wgpu_groups_morph.rs`, and
    // the blend itself (on the encoded bytes, as three.js's canvas does) in
    // `tests/support/blending.rs` on both backends.
    let mesh = Mesh::with_materials(
        std::sync::Arc::new(g),
        vec![flat(0x00ff00), flat(0xff0000), flat(0x0000ff)],
    );
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::mesh(mesh));
    let a = m.render_to_rgba(&mut scene, &cam).unwrap();
    let b = w.render_to_rgba(&mut scene, &cam);
    let (differ, lit_px) = colour_diff(&a, &b, 3);
    eprintln!("groups parity: {lit_px} lit, {differ} differ");
    assert!(lit_px > 800);
    assert!(differ * 50 <= lit_px, "{differ} of {lit_px} pixels differ");
}

/// Morph targets (absolute and relative, positions and normals, lit) and a
/// skinned mesh with morph targets: the two backends agree on coverage and
/// shading.
#[test]
fn morph_targets_match_the_wgpu_backend() {
    use threers::core::ObjectKind;
    let Some(mut m) = renderer(1) else { return };
    let Some(mut w) = wgpu_renderer() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let cam = unit_camera();
    for relative in [false, true] {
        let mut g = PlaneGeometry::with_segments(1.0, 1.0, 4, 4);
        let base = g.get_attribute("position").unwrap().array.clone();
        let (mut t0, mut n0) = (Vec::new(), Vec::new());
        let (s, c) = (40f32.to_radians().sin(), 40f32.to_radians().cos());
        for p in base.chunks_exact(3) {
            let target = [p[0] * 1.5, p[1] + 0.2 * p[0], p[2]];
            if relative {
                t0.extend_from_slice(&[target[0] - p[0], target[1] - p[1], 0.0]);
                n0.extend_from_slice(&[s, 0.0, c - 1.0]);
            } else {
                t0.extend_from_slice(&target);
                n0.extend_from_slice(&[s, 0.0, c]);
            }
        }
        g.morph_targets_relative = relative;
        g.morph_attributes.insert("normal".into(), vec![BufferAttribute::new(n0, 3)]);
        g.set_morph_attribute("position", vec![BufferAttribute::new(t0, 3)]);
        // Standard rather than Lambert: the two backends' Lambert paths are
        // not held to each other (Metal's runs about 3x brighter), the
        // physical one is (`tests/backend_parity.rs`).
        let mut mat = threers::materials::StandardMaterial::new(Color::WHITE);
        mat.metalness = 0.0;
        mat.roughness = 1.0;
        let mut mesh = Mesh::new(g, mat.into());
        mesh.morph_influences = vec![0.7];
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        scene.add(Object3D::mesh(mesh));
        let mut light = Object3D::light(DirectionalLight::new(Color::WHITE, 1.0));
        light.position = Vector3::new(0.5, 0.0, 1.0);
        scene.add(light);
        let a = m.render_to_rgba(&mut scene, &cam).unwrap();
        let b = w.render_to_rgba(&mut scene, &cam);
        let (cov, lit_a, lit_b) = coverage_diff(&a, &b);
        let (shade, _) = colour_diff(&a, &b, 12);
        eprintln!("morph relative={relative}: metal {lit_a}, wgpu {lit_b}, coverage differs {cov}, shading differs {shade}");
        assert!(lit_a > 1000);
        assert!(cov * 50 <= lit_a, "coverage: {cov} of {lit_a}");
        assert!(shade * 20 <= lit_a, "shading: {shade} of {lit_a}");
    }

    // Skinned + morph.
    let (mut scene, id) = skinned_scene();
    if let ObjectKind::SkinnedMesh(sm) = &mut scene.arena.get_mut(id).unwrap().kind {
        let mut g = (*sm.geometry).clone();
        let n = g.get_attribute("position").unwrap().count();
        let pos = g.get_attribute("position").unwrap().array.clone();
        let delta: Vec<f32> = (0..n).flat_map(|v| [pos[v * 3] * 0.6, pos[v * 3 + 1] * 0.4, 0.0]).collect();
        g.morph_targets_relative = true;
        g.set_morph_attribute("position", vec![BufferAttribute::new(delta, 3)]);
        sm.geometry = std::sync::Arc::new(g);
        sm.update_morph_targets();
        sm.morph_influences = vec![0.9];
    }
    let a = m.render_to_rgba(&mut scene, &cam).unwrap();
    let b = w.render_to_rgba(&mut scene, &cam);
    let (cov, lit_a, lit_b) = coverage_diff(&a, &b);
    eprintln!("skinned morph: metal {lit_a}, wgpu {lit_b}, {cov} differ");
    assert!(lit_a > 150);
    assert!(cov * 40 <= lit_a, "{cov} of {lit_a}");
}

#[path = "support/instanced_truth.rs"]
mod instanced_truth;

fn truth_renderer() -> Option<MetalHeadlessRenderer> {
    match MetalHeadlessRenderer::builder()
        .size(instanced_truth::N, instanced_truth::N)
        .build()
    {
        Ok(r) => Some(r),
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            None
        }
        Err(e) => panic!("Metal renderer could not be built: {e}"),
    }
}

/// `InstancedMesh` with a material array (per-group draws, each instanced)
/// and morph targets — shared influences and three.js's per-instance
/// `setMorphAt` rows — drawn as three.js draws the equivalent plain meshes,
/// and picked with each instance's `materialIndex`.
#[test]
fn instanced_meshes_draw_groups_and_morphs_as_three_js_does() {
    let Some(mut m) = truth_renderer() else { return };
    let failures = instanced_truth::check("metal", &mut |scene, cam| {
        m.render_to_rgba(scene, cam).expect("metal render")
    });
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// three.js morphs a 4-component vertex colour in all four channels.
#[test]
fn four_component_colours_morph_with_their_alpha() {
    let Some(mut m) = truth_renderer() else { return };
    let failures = instanced_truth::check_colour_morph("metal", &mut |scene, cam| {
        m.render_to_rgba(scene, cam).expect("metal render")
    });
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// three.js `instanceColor` multiplies the diffuse colour (and any vertex
/// colour) per instance.
#[test]
fn instance_colours_tint_each_instance_as_three_js_does() {
    let Some(mut m) = truth_renderer() else { return };
    let failures = instanced_truth::check_instance_colours("metal", &mut |scene, cam| {
        m.render_to_rgba(scene, cam).expect("metal render")
    });
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[path = "support/polygon_offset.rs"]
mod decal;

/// three.js `polygonOffset` keeps a decal over the surface it was cut from,
/// through `setDepthBias:slopeScale:clamp:` — and without it the two z-fight.
#[test]
fn polygon_offset_keeps_a_decal_over_its_surface() {
    for msaa in [1, 4] {
        let mut m = match MetalHeadlessRenderer::builder().size(decal::N, decal::N).msaa(msaa).build() {
            Ok(m) => m,
            Err(MetalError::NoDevice) => return,
            Err(e) => panic!("{e}"),
        };
        let [(blue0, red0), (blue1, red1)] =
            decal::measure(&mut |s, c| m.render_to_rgba(s, c).expect("metal render"));
        eprintln!("metal msaa {msaa}: no offset {blue0} blue / {red0} red; offset {blue1} blue / {red1} red");
        assert!(blue1 > 1000, "msaa {msaa}: decal interior is {blue1} pixels");
        assert_eq!(red1, 0, "msaa {msaa}: with polygonOffset the surface shows through the decal");
        assert!(red0 * 20 > blue0 + red0, "msaa {msaa}: without an offset the two should z-fight");
    }
}

#[path = "support/alpha_test.rs"]
mod alpha;

/// three.js `alphaTest` and `transparent` on every material kind, held to the
/// same three.js rules as the wgpu renderer (`tests/alpha_test.rs`).
#[test]
fn alpha_test_and_transparent_follow_three_js() {
    let mut failures = Vec::new();
    for msaa in [1, 4] {
        let mut m = match MetalHeadlessRenderer::builder().size(alpha::N, alpha::N).msaa(msaa).build() {
            Ok(m) => m,
            Err(MetalError::NoDevice) => return,
            Err(e) => panic!("{e}"),
        };
        failures.extend(alpha::check(&format!("metal msaa {msaa}"), &mut |s, c| {
            m.render_to_rgba(s, c).expect("metal render")
        }));
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

#[path = "support/shadow_cutout.rs"]
mod cutout;

/// A caster cut by `alphaTest` and its map casts only what is left of it, and
/// a caster's `side` picks the faces its shadow records (three.js's
/// `shadowSide`) — directional, spot and point, as on wgpu.
#[test]
fn alpha_tested_casters_cast_cut_out_shadows() {
    let mut m = match MetalHeadlessRenderer::builder().size(cutout::N, cutout::N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("{e}"),
    };
    let failures = cutout::check("metal", &mut |s, c| m.render_to_rgba(s, c).expect("metal render"));
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

#[path = "support/lit_maps.rs"]
mod lit_maps;

/// three.js's `map`, `aoMap` and `emissiveMap` on Lambert, Phong and Toon, as
/// on wgpu (`tests/lit_maps.rs`).
#[test]
fn lambert_phong_toon_shade_their_maps_as_three_js_does() {
    let mut m = match MetalHeadlessRenderer::builder().size(lit_maps::N, lit_maps::N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("{e}"),
    };
    let failures = lit_maps::check("metal", &mut |s, c| m.render_to_rgba(s, c).expect("metal render"));
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

#[path = "support/material_variants.rs"]
mod variants;

/// Each texture keeps its own uv transform and sampler state — mirrored
/// repeat included — as on wgpu (`tests/material_variants.rs`).
#[test]
fn each_texture_keeps_its_own_sampler_and_transform() {
    let mut m = match MetalHeadlessRenderer::builder().size(variants::N, variants::N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("{e}"),
    };
    let failures = variants::check("metal", &mut |s, c| m.render_to_rgba(s, c).expect("metal render"));
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

#[path = "support/env_map.rs"]
mod env_map;

/// three.js `envMap` on Basic, Lambert and Phong, as on wgpu
/// (`tests/env_map.rs`).
#[test]
fn env_maps_reflect_and_combine_as_three_js_does() {
    let mut m = match MetalHeadlessRenderer::builder().size(env_map::N, env_map::N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("{e}"),
    };
    let failures = env_map::check("metal", &mut |s, c| m.render_to_rgba(s, c).expect("metal render"));
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

#[path = "support/points.rs"]
mod points;

/// three.js `Points`, as on wgpu (`tests/points.rs`).
#[test]
fn points_draw_as_three_js_does() {
    let mut m = match MetalHeadlessRenderer::builder().size(points::N, points::N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("{e}"),
    };
    let failures = points::check("metal", &mut |s, c| m.render_to_rgba(s, c).expect("metal render"));
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

#[path = "support/pbr_env.rs"]
mod pbr_env;

/// three.js `envMap`, `envMapIntensity`, `envMapRotation` and the scene's
/// `environmentIntensity` / `environmentRotation`, as on wgpu
/// (`tests/pbr_env.rs`).
#[test]
fn standard_materials_read_their_environment_as_three_js_does() {
    let mut m = match MetalHeadlessRenderer::builder().size(pbr_env::N, pbr_env::N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("{e}"),
    };
    let failures = pbr_env::check("metal", &mut |s, c| m.render_to_rgba(s, c).expect("metal render"));
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

#[path = "support/physical_specular.rs"]
mod physical_specular;

/// MeshPhysicalMaterial's `ior`, `specularIntensity` and `specularColor`
/// against three.js r165's formulas, as on wgpu (`tests/physical.rs`).
#[test]
fn ior_and_specular_follow_three_js() {
    let mut m = match MetalHeadlessRenderer::builder().size(physical_specular::N, physical_specular::N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("{e}"),
    };
    let failures = physical_specular::check("metal", &mut |s, c| m.render_to_rgba(s, c).expect("metal render"));
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

#[path = "support/physical_maps.rs"]
mod physical_maps;

/// Each MeshPhysicalMaterial map draws what its scalar does with the texel
/// folded in, as on wgpu (`tests/physical.rs`).
#[test]
fn physical_maps_scale_their_scalars_as_three_js_does() {
    let mut m = match MetalHeadlessRenderer::builder().size(physical_maps::N, physical_maps::N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("{e}"),
    };
    let failures = physical_maps::check("metal", &mut |s, c| m.render_to_rgba(s, c).expect("metal render"));
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// Transmission seen in stereo: each eye refracts the scene captured from
/// where *it* stands. Drawn layered (a slice per eye, one pass) and side by
/// side (two viewports of one texture, a pass per eye), each eye's picture
/// must be the mono render from that eye's camera — the capture from the
/// other eye shifts what shows through the glass.
#[test]
fn each_eye_refracts_its_own_capture() {
    use threers::cameras::PerspectiveCamera;
    use threers::geometries::SphereGeometry;
    use threers::materials::PhysicalMaterial;
    use threers::metal::enums::MTLViewport;

    let Ok(device) = MetalDevice::new() else {
        return;
    };
    let mut renderer = threers::metal::MetalRenderer::with_device(device.clone()).unwrap();
    const S: u32 = 96;
    let format = threers::metal::enums::pixel_format::RGBA8_UNORM_SRGB;

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add_light(DirectionalLight::new(Color::WHITE, 2.0).with_direction(Vector3::new(-1.0, -1.0, -2.0)));
    // Bars of different colours behind the glass, so a shifted capture shows.
    for (i, hex) in [0xd02020u32, 0x20d020, 0x2020d0, 0xd0d020, 0xd020d0].into_iter().enumerate() {
        let mut m = StandardMaterial::new(Color::from_hex(hex));
        m.emissive = Color::from_hex(hex);
        let mut bar = Object3D::mesh(Mesh::new(BoxGeometry::new(0.5, 5.0, 0.2), m.into()));
        bar.position = Vector3::new(-2.0 + i as f32, 0.0, -2.5);
        scene.add(bar);
    }
    let mut glass = PhysicalMaterial::new(Color::WHITE);
    glass.roughness = 0.05;
    glass.transmission = 1.0;
    glass.thickness = 1.5;
    glass.ior = 1.5;
    scene.add(Object3D::mesh(Mesh::new(SphereGeometry::new(1.0, 48, 24), glass.into())));

    let camera = |x: f32| {
        let mut c = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
        c.position = Vector3::new(x, 0.0, 4.0);
        c.look_at(Vector3::new(x * 0.5, 0.0, 0.0));
        c
    };
    let eyes = [camera(-0.8), camera(0.8)];
    let mono: Vec<Vec<u8>> = eyes
        .iter()
        .map(|c| {
            let t = MetalRenderTarget::new(&device, S, S, format, 1).unwrap();
            renderer.render(&mut scene, c, &t.attachments()).unwrap();
            t.read_rgba(&device).unwrap()
        })
        .collect();
    let worst = |a: &[u8], b: &[u8]| {
        a.chunks(4)
            .zip(b.chunks(4))
            .map(|(p, q)| (0..3).map(|k| (p[k] as i32 - q[k] as i32).abs()).max().unwrap())
            .max()
            .unwrap()
    };
    // The two eyes do see different things through the glass.
    assert!(worst(&mono[0], &mono[1]) > 60, "the eyes' mono renders barely differ");

    // Layered: a slice per eye.
    let target = MetalRenderTarget::layered(&device, S, S, format, 2).unwrap();
    let views = [
        RenderView::from_camera(&eyes[0]).with_slice(0),
        RenderView::from_camera(&eyes[1]).with_slice(1),
    ];
    renderer.render_views(&mut scene, &views, &target.attachments()).unwrap();
    for (i, m) in mono.iter().enumerate() {
        let got = target.read_rgba_slice(&device, i as u32).unwrap();
        let d = worst(&got, m);
        assert!(d <= 2, "layered eye {i} is {d} levels from its mono render");
    }

    // Side by side: two viewports, a pass each.
    let wide = MetalRenderTarget::new(&device, 2 * S, S, format, 1).unwrap();
    let half = |x: f64| MTLViewport { origin_x: x, origin_y: 0.0, width: S as f64, height: S as f64, znear: 0.0, zfar: 1.0 };
    let views = [
        RenderView::from_camera(&eyes[0]).with_viewport(half(0.0)),
        RenderView::from_camera(&eyes[1]).with_viewport(half(S as f64)),
    ];
    renderer.render_views(&mut scene, &views, &wide.attachments()).unwrap();
    let px = wide.read_rgba(&device).unwrap();
    for (i, m) in mono.iter().enumerate() {
        let got: Vec<u8> = (0..S as usize)
            .flat_map(|y| {
                let row = (y * 2 * S as usize + i * S as usize) * 4;
                px[row..row + S as usize * 4].to_vec()
            })
            .collect();
        let d = worst(&got, m);
        assert!(d <= 2, "side-by-side eye {i} is {d} levels from its mono render");
    }
}

#[path = "support/toon_gradient.rs"]
mod toon_gradient;

/// MeshToonMaterial's `gradientMap`, as on wgpu (`tests/toon.rs`).
#[test]
fn toon_gradient_map_shades_as_three_js_does() {
    let mut m = match MetalHeadlessRenderer::builder().size(toon_gradient::N, toon_gradient::N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("{e}"),
    };
    let failures = toon_gradient::check("metal", &mut |s, c| m.render_to_rgba(s, c).expect("metal render"));
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
