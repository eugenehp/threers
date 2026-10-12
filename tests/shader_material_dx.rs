//! A broken `ShaderMaterial` has to be findable, and findable without a GPU.
//!
//! The module a custom shader actually compiles as is the built-in preamble —
//! about seventeen hundred lines of it — plus the user's forty. So the error
//! wgpu reports names a line the caller has never seen, and it arrives as a
//! process-ending panic from inside the draw loop rather than as a value.
//! These check the three things that fixes: it is caught, it is explained, and
//! it does not have to happen at render time.

use threers::{
    BoxGeometry, Color, HeadlessRenderer, Material, Mesh, Object3D, PerspectiveCamera, Scene,
    ShaderMaterial, Vector3,
};

const GOOD: &str = "@fragment\n\
     fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
     return vec4<f32>(in.uv.x, in.uv.y, 0.0, 1.0);\n\
     }\n";

#[test]
fn a_good_shader_validates_without_a_gpu() {
    let m = ShaderMaterial::new(GOOD);
    assert!(m.validate(true).is_ok(), "{:?}", m.validate(true));
    assert!(m.validate(false).is_ok(), "and on a device with no storage");
}

#[test]
fn a_typo_is_reported_rather_than_panicked() {
    let m = ShaderMaterial::new(
        "@fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
         return vec4<f32>(in.uv.x, nope, 0.0, 1.0);\n\
         }\n",
    );
    let err = m.validate(true).expect_err("a typo should not validate");
    assert!(
        err.contains("nope"),
        "the message should name the identifier: {err}"
    );
    assert!(
        err.contains("preamble"),
        "the message should say where the line numbers are from: {err}"
    );
}

/// `u_s0` only exists where the device has storage buffers, and `s0` exists
/// everywhere. That is exactly the kind of thing worth finding before shipping
/// to a browser, and `validate(false)` is how.
#[test]
fn validate_tells_the_two_backends_apart() {
    let raw = ShaderMaterial::new(
        "@fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
         return vec4<f32>(u_s0[0], 0.0, 0.0, 1.0);\n\
         }\n",
    );
    assert!(raw.validate(true).is_ok(), "storage buffers are present");
    assert!(
        raw.validate(false).is_err(),
        "`u_s0` cannot exist on a device with no storage buffers"
    );

    let portable = ShaderMaterial::new(
        "@fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
         return vec4<f32>(s0(0u), 0.0, 0.0, 1.0);\n\
         }\n",
    );
    assert!(portable.validate(true).is_ok());
    assert!(portable.validate(false).is_ok(), "`s0` works on both");
}

/// A custom vertex stage is validated too, and against the right entry name.
#[test]
fn a_custom_vertex_stage_is_validated() {
    let ok = ShaderMaterial::new(GOOD).with_vertex(
        "@vertex\n\
         fn vs_custom(in: VsIn) -> VsOut { return standard_vertex(in); }\n",
    );
    assert!(ok.validate(true).is_ok(), "{:?}", ok.validate(true));

    let bad = ShaderMaterial::new(GOOD).with_vertex(
        "@vertex\n\
         fn vs_custom(in: VsIn) -> VsOut { return nonsense(in); }\n",
    );
    assert!(bad.validate(true).is_err());
}

/// And at render time a bad shader costs one object, not the process.
#[test]
fn a_bad_shader_skips_its_draw_instead_of_ending_the_program() {
    let Ok(mut r) = HeadlessRenderer::builder()
        .size(32, 32)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
    else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    // One material that cannot compile, and one that can, in the same frame.
    scene.add_mesh(
        BoxGeometry::new(4.0, 4.0, 0.1),
        Material::Shader(ShaderMaterial::new(
            "@fragment\n\
             fn fs_main(in: VsOut) -> @location(0) vec4<f32> { return syntax error; }\n",
        )),
    );
    let mut good = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 0.1),
        Material::Shader(ShaderMaterial::new(GOOD)),
    ));
    good.position = Vector3::new(0.0, 0.0, 1.0);
    scene.add(good);

    let mut cam = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 3.0);
    cam.look_at(Vector3::ZERO);
    scene.update_world();

    // Two frames: the second proves the failure is remembered rather than
    // retried, and that neither one takes the process down.
    let px = r.render_to_rgba(&mut scene, &cam);
    let _ = r.render_to_rgba(&mut scene, &cam);
    let centre = ((16 * 32 + 16) * 4) as usize;
    assert!(
        px[centre] > 0 || px[centre + 1] > 0,
        "the material that does compile should still have drawn"
    );
}

/// The fov trap, closed. `new` takes degrees and the field is radians, so
/// `camera.fov.to_radians()` — which reads correctly — is wrong by 57x and
/// fails silently.
#[test]
fn the_camera_reports_its_field_of_view_in_both_units() {
    let cam = PerspectiveCamera::new(34.0, 1.0, 0.1, 100.0);
    assert!((cam.fov_degrees() - 34.0).abs() < 1e-3);
    assert!((cam.fov - 34.0_f32.to_radians()).abs() < 1e-6);

    // And the number anything sub-pixel actually needs.
    let ppu = cam.pixels_per_unit(5.0, 1000.0);
    let expect = 1000.0 * 0.5 / (34.0_f32.to_radians() * 0.5).tan() / 5.0;
    assert!((ppu - expect).abs() < 1e-2, "{ppu} against {expect}");
}
