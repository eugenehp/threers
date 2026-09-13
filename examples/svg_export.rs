//! Export a rendered scene to SVG.
//!
//! ```text
//! cargo run --example svg_export
//! ```
//!
//! Writes four files to `out/`, one per mode, so the trade-offs are visible
//! side by side:
//!
//! | File | What it is |
//! |------|------------|
//! | `svg_export_lit.svg` | Lambert shading from the scene's lights |
//! | `svg_export_flat.svg` | Unlit material colour — the flat-vector look |
//! | `svg_export_wireframe.svg` | Triangle edges only |
//! | `svg_export_ortho.svg` | Lit, through an orthographic camera |
//!
//! Everything here runs on the CPU: no GPU, no window, no adapter. That is the
//! other reason to reach for this renderer — it works in CI, over SSH, and in
//! a wasm build with no WebGPU.

use threers::prelude::*;
use threers::{OrthographicCamera, SphereGeometry, TorusKnotGeometry};

fn main() -> std::io::Result<()> {
    let mut scene = build_scene();

    let mut camera = PerspectiveCamera::new(45.0, 4.0 / 3.0, 0.1, 100.0);
    camera.position = Vector3::new(4.0, 3.0, 6.0);
    camera.look_at(Vector3::ZERO);

    let renderer = SvgRenderer::new(1200, 900);
    renderer.render_to_file(&mut scene, &camera, "out/svg_export_lit.svg")?;

    for (shading, name) in [
        (SvgShading::Flat, "out/svg_export_flat.svg"),
        (SvgShading::Wireframe, "out/svg_export_wireframe.svg"),
    ] {
        let options = SvgOptions {
            shading,
            ..SvgOptions::default()
        };
        SvgRenderer::new(1200, 900)
            .with_options(options)
            .render_to_file(&mut scene, &camera, name)?;
    }

    // An orthographic camera is where vector output earns its keep: no
    // perspective divide, so the result is a true scale drawing.
    let mut ortho = OrthographicCamera::new(-5.0, 5.0, 3.75, -3.75, 0.1, 100.0);
    ortho.position = Vector3::new(4.0, 3.0, 6.0);
    ortho.target = Vector3::ZERO;
    renderer.render_to_file(&mut scene, &ortho, "out/svg_export_ortho.svg")?;

    for f in [
        "out/svg_export_lit.svg",
        "out/svg_export_flat.svg",
        "out/svg_export_wireframe.svg",
        "out/svg_export_ortho.svg",
    ] {
        println!("{f}  {} KB", std::fs::metadata(f)?.len() / 1024);
    }
    Ok(())
}

fn build_scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0xf4f1ea);

    let mut knot = Object3D::mesh(Mesh::new(
        TorusKnotGeometry::new(1.2, 0.38, 160, 24, 2, 3),
        Material::Lambert(LambertMaterial::new(Color::from_hex(0xd4553c))),
    ));
    // The knot reaches 2.076 below its own origin, so this is what rests it on
    // the floor at y = -1.5 with a little clearance. Sunk into the plane it
    // renders with a flat-cut bottom, which is correct and looks like a bug.
    knot.position = Vector3::new(0.0, 0.65, 0.0);
    scene.add(knot);

    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.9, 48, 24),
        Material::Lambert(LambertMaterial::new(Color::from_hex(0x2f6f8f))),
    ));
    ball.position = Vector3::new(2.4, -0.5, -1.2);
    scene.add(ball);

    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(14.0, 14.0),
        Material::Lambert(LambertMaterial::new(Color::from_hex(0xe8e2d5))),
    ));
    floor.position = Vector3::new(0.0, -1.5, 0.0);
    floor.quaternion =
        Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), -std::f32::consts::FRAC_PI_2);
    scene.add(floor);

    scene.add_light(AmbientLight::new(Color::from_hex(0xb9c6d6), 0.35));
    scene.add_light(HemisphereLight::new(
        Color::from_hex(0xcfe3f5),
        Color::from_hex(0x4a4036),
        0.4,
    ));

    // A directional light points from its position at the origin, so placing it
    // is how you aim it.
    let mut key = Object3D::light(DirectionalLight::new(Color::from_hex(0xfff2dd), 3.2));
    key.position = Vector3::new(5.0, 7.0, 4.0);
    scene.add(key);

    let mut rim = Object3D::light(DirectionalLight::new(Color::from_hex(0x9ec5ff), 1.4));
    rim.position = Vector3::new(-6.0, 2.0, -5.0);
    scene.add(rim);

    scene
}
