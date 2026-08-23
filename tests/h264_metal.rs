//! Metal headless readback → native H.264/MP4.
//!
//! The encoder is CPU/wasm; this proves the Apple GPU path that feeds it.
//! Skips when the `metal` feature is off, the target is not Apple, or no device.
//!
//! ```text
//! cargo test --features "metal,native-codec" --test h264_metal -- --nocapture
//! ```
#![cfg(all(
    feature = "metal",
    feature = "native-codec",
    any(target_os = "macos", target_os = "ios")
))]

use threers::metal::{MetalError, MetalHeadlessRenderer};
use threers::{
    encode_animation_rgba, AnimationEncodeOptions, BasicMaterial, BrowserCodec, Color, Material,
    Mesh, Object3D, OrthographicCamera, PlaneGeometry, Scene, Vector3,
};

const W: u32 = 64;
const H: u32 = 48;

fn renderer() -> Option<MetalHeadlessRenderer> {
    match MetalHeadlessRenderer::builder().size(W, H).msaa(1).build() {
        Ok(r) => Some(r),
        Err(MetalError::NoDevice) => {
            eprintln!("skipping h264_metal: no Metal device");
            None
        }
        Err(e) => panic!("Metal renderer could not be built: {e}"),
    }
}

fn unit_camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 2.0);
    cam.target = Vector3::ZERO;
    cam
}

#[test]
fn metal_solid_frames_encode_mp4() {
    let Some(mut hr) = renderer() else { return };

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x112233);
    let mesh = scene.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(2.0, 2.0),
        Material::Basic(BasicMaterial::new(Color::from_hex(0x44aaff))),
    )));
    let cam = unit_camera();

    let mut frames = Vec::with_capacity(4);
    for i in 0..4 {
        if let Some(obj) = scene.get_mut(mesh) {
            obj.rotate_y(0.4);
        }
        frames.push(hr.render_to_rgba(&mut scene, &cam).expect("Metal readback"));
        assert_eq!(frames[i].len(), (W * H * 4) as usize);
    }

    let mp4 = encode_animation_rgba(
        &AnimationEncodeOptions {
            width: W,
            height: H,
            fps: 12,
            codec: BrowserCodec::Mp4,
            transparent: false,
            gif_colors: 256,
        },
        frames,
    )
    .expect("metal rgba → mp4");
    assert_eq!(&mp4[4..8], b"ftyp");
    assert!(mp4.len() > 200, "mp4 too small: {}", mp4.len());
    eprintln!("Metal H.264 MP4 OK: {W}x{H} x4 ({} bytes)", mp4.len());
}
