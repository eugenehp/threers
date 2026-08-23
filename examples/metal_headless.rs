//! Offscreen rendering through the Metal backend.
//!
//! ```sh
//! cargo run --release --features metal --example metal_headless
//! ```
//!
//! Writes `out/metal_headless.png`. No window, no main thread requirement, no
//! wgpu — the frame is drawn by `src/metal`, which talks to Metal through the
//! Objective-C runtime directly.

use std::path::PathBuf;
use std::sync::Arc;

use threers::core::{InstancedMesh, Mesh, Object3D};
use threers::geometries::{BoxGeometry, PlaneGeometry, SphereGeometry};
use threers::lights::{AmbientLight, DirectionalLight, PointLight};
use threers::materials::{BasicMaterial, Material, StandardMaterial};
use threers::math::{Color, Matrix4, Quaternion, Vector3};
use threers::metal::{MetalError, MetalHeadlessRenderer};
use threers::scene::Scene;
use threers::textures::{Texture, TextureFilter, TextureFormat, TextureWrap};
use threers::{encode_png, PerspectiveCamera};

const W: u32 = 1280;
const H: u32 = 720;

fn main() -> Result<(), MetalError> {
    let mut renderer = MetalHeadlessRenderer::builder()
        .size(W, H)
        .msaa(4)
        .build()?;
    println!(
        "device: {} ({} memory)",
        renderer.device().name(),
        if renderer.device().has_unified_memory() {
            "unified"
        } else {
            "discrete"
        }
    );

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0d1117);

    // A checkered floor, to show the base-colour map path and UV wrapping.
    let mut floor_texture = Texture::new(
        2,
        2,
        TextureFormat::Rgba8UnormSrgb,
        vec![
            210, 210, 214, 255, 40, 44, 52, 255, //
            40, 44, 52, 255, 210, 210, 214, 255,
        ],
    );
    floor_texture.mag_filter = TextureFilter::Nearest;
    floor_texture.min_filter = TextureFilter::Nearest;
    floor_texture.wrap_s = TextureWrap::Repeat;
    floor_texture.wrap_t = TextureWrap::Repeat;
    floor_texture.repeat = threers::math::Vector2::new(12.0, 12.0);

    let mut floor_material = BasicMaterial::new(Color::WHITE);
    floor_material.map = Some(Arc::new(floor_texture));
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(24.0, 24.0),
        Material::Basic(floor_material),
    ));
    floor.quaternion =
        Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), -std::f32::consts::FRAC_PI_2);
    floor.position = Vector3::new(0.0, -1.0, 0.0);
    scene.add(floor);

    // A rough dielectric and a polished metal, side by side.
    let mut sphere = Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 48, 32),
        Material::Standard(
            StandardMaterial::new(Color::from_hex(0xff7043))
                .with_roughness(0.35)
                .with_metalness(0.0),
        ),
    ));
    sphere.position = Vector3::new(-1.6, 0.0, 0.0);
    scene.add(sphere);

    let mut metal_sphere = Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 48, 32),
        Material::Standard(
            StandardMaterial::new(Color::from_hex(0xb0bec5))
                .with_roughness(0.15)
                .with_metalness(1.0),
        ),
    ));
    metal_sphere.position = Vector3::new(1.6, 0.0, 0.0);
    scene.add(metal_sphere);

    // A row of instanced cubes: one draw call, one uploaded geometry.
    let count = 9;
    let mut cubes = InstancedMesh::new(
        BoxGeometry::new(0.35, 0.35, 0.35),
        Material::Standard(StandardMaterial::new(Color::from_hex(0x4fc3f7)).with_roughness(0.5)),
        count,
    );
    for i in 0..count {
        let t = i as f32 / (count - 1) as f32;
        let x = (t - 0.5) * 7.0;
        cubes.set_matrix_at(
            i,
            Matrix4::compose(
                Vector3::new(x, -0.6 + (t * std::f32::consts::PI).sin() * 0.5, -2.2),
                Quaternion::from_axis_angle(Vector3::UP, t * 2.0),
                Vector3::ONE,
            ),
        );
    }
    scene.add(Object3D::instanced_mesh(cubes));

    scene.add_light(AmbientLight::new(Color::from_hex(0x30506a), 1.0));
    let mut key = Object3D::light(DirectionalLight::new(Color::from_hex(0xfff3e0), 2.6));
    key.position = Vector3::new(4.0, 6.0, 5.0);
    scene.add(key);
    let mut rim = Object3D::light(PointLight::new(Color::from_hex(0x64b5f6), 30.0));
    rim.position = Vector3::new(-4.0, 1.5, -3.0);
    scene.add(rim);

    let mut camera = PerspectiveCamera::new(45.0, W as f32 / H as f32, 0.1, 200.0);
    camera.position = Vector3::new(0.0, 1.8, 7.5);
    camera.look_at(Vector3::new(0.0, 0.0, 0.0));

    let rgba = renderer.render_to_rgba(&mut scene, &camera)?;
    let stats = renderer.stats();
    println!(
        "{} draw calls, {} triangles, {} geometries + {} textures uploaded",
        stats.draw_calls, stats.triangles, stats.geometry_uploads, stats.texture_uploads
    );

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out).expect("create out/");
    let path = out.join("metal_headless.png");
    std::fs::write(&path, encode_png(W, H, &rgba)).expect("write png");
    println!("wrote {}", path.display());
    Ok(())
}
