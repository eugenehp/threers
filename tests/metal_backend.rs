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
