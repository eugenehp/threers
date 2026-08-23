//! Stereo rendering through the Metal backend — the visionOS frame shape,
//! offscreen so it can be run on a Mac.
//!
//! ```sh
//! cargo run --release --features metal --example metal_stereo
//! ```
//!
//! Two eyes, 63 mm apart, drawn in **one pass** into the two slices of a texture
//! array — which is exactly what `MetalRenderer::render_views` does with the
//! layered attachments the visionOS compositor hands over. Here the slices are
//! read back and written side by side, so you can cross your eyes at
//! `out/metal_stereo.png` and see the parallax the headset would.

use std::path::PathBuf;

use threers::metal::enums::pixel_format;
use threers::metal::{MetalDevice, MetalError, MetalRenderTarget, MetalRenderer, RenderView};
use threers::prelude::*;

const EYE_W: u32 = 640;
const EYE_H: u32 = 640;
/// Interpupillary distance, metres. The human average, and roughly what a
/// Vision Pro reports.
const IPD: f32 = 0.063;

fn main() -> Result<(), MetalError> {
    let device = MetalDevice::new()?;
    println!("device: {}", device.name());
    let mut renderer = MetalRenderer::with_device(device.clone())?;

    // The compositor's `layered` layout: one colour array, one depth array, a
    // slice per eye.
    let target =
        MetalRenderTarget::layered(&device, EYE_W, EYE_H, pixel_format::RGBA8_UNORM_SRGB, 2)?;

    let mut scene = build_scene();

    // Both eyes look the same way from either side of the head. A headset's
    // matrices arrive from `cp_view_get_transform` instead of being built here,
    // but they mean the same thing.
    let head = Vector3::new(0.0, 0.25, 1.2);
    let target_point = Vector3::new(0.0, 0.0, -0.6);
    let eye = |offset: f32, slice: u32| {
        let mut camera = PerspectiveCamera::new(70.0, EYE_W as f32 / EYE_H as f32, 0.01, 100.0);
        camera.position = Vector3::new(head.x + offset, head.y, head.z);
        camera.target = Vector3::new(
            target_point.x + offset * 0.15,
            target_point.y,
            target_point.z,
        );
        RenderView::from_camera(&camera).with_slice(slice)
    };
    let views = [eye(-IPD * 0.5, 0), eye(IPD * 0.5, 1)];

    let stats = renderer.render_views(&mut scene, &views, &target.attachments())?;
    println!(
        "{} draw calls for {} eyes, {} triangles",
        stats.draw_calls,
        views.len(),
        stats.triangles
    );

    let left = target.read_rgba_slice(&device, 0)?;
    let right = target.read_rgba_slice(&device, 1)?;
    let side_by_side = join_horizontally(&left, &right, EYE_W, EYE_H);

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out).expect("create out/");
    let path = out.join("metal_stereo.png");
    std::fs::write(&path, encode_png(EYE_W * 2, EYE_H, &side_by_side)).expect("write png");
    println!("wrote {}", path.display());
    Ok(())
}

/// Objects at three depths, so the parallax between the eyes is obvious.
fn build_scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0b1020);

    let mut near = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.12, 48, 32),
        Material::Standard(StandardMaterial::new(Color::from_hex(0xff7043)).with_roughness(0.3)),
    ));
    near.position = Vector3::new(-0.25, 0.05, 0.15);
    scene.add(near);

    let mut mid = Object3D::mesh(Mesh::new(
        TorusKnotGeometry::new(0.22, 0.07, 160, 24, 2, 3),
        Material::Standard(
            StandardMaterial::new(Color::from_hex(0x4fc3f7))
                .with_roughness(0.25)
                .with_metalness(0.2),
        ),
    ));
    mid.position = Vector3::new(0.1, 0.1, -0.6);
    scene.add(mid);

    let mut far = Object3D::mesh(Mesh::new(
        BoxGeometry::new(0.6, 0.6, 0.6),
        Material::Standard(StandardMaterial::new(Color::from_hex(0x81c784)).with_roughness(0.7)),
    ));
    far.position = Vector3::new(-0.1, 0.0, -2.4);
    far.quaternion = Quaternion::from_axis_angle(Vector3::UP, 0.6);
    scene.add(far);

    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(12.0, 12.0),
        Material::Standard(StandardMaterial::new(Color::from_hex(0x1b2430)).with_roughness(0.9)),
    ));
    floor.quaternion =
        Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), -std::f32::consts::FRAC_PI_2);
    floor.position = Vector3::new(0.0, -0.4, 0.0);
    scene.add(floor);

    scene.add_light(AmbientLight::new(Color::from_hex(0x2a3a52), 1.0));
    let mut key = Object3D::light(DirectionalLight::new(Color::from_hex(0xfff0dd), 2.6));
    key.position = Vector3::new(2.0, 3.0, 2.0);
    scene.add(key);
    let mut fill = Object3D::light(PointLight::new(Color::from_hex(0x5c9dff), 6.0));
    fill.position = Vector3::new(-1.5, 0.6, 0.4);
    scene.add(fill);

    scene
}

/// Put two equally sized RGBA images side by side.
fn join_horizontally(left: &[u8], right: &[u8], width: u32, height: u32) -> Vec<u8> {
    let row = width as usize * 4;
    let mut out = vec![0u8; row * 2 * height as usize];
    for y in 0..height as usize {
        out[y * row * 2..y * row * 2 + row].copy_from_slice(&left[y * row..(y + 1) * row]);
        out[y * row * 2 + row..(y + 1) * row * 2].copy_from_slice(&right[y * row..(y + 1) * row]);
    }
    out
}
