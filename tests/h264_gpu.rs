//! GPU readback → native H.264/MP4 (wgpu `HeadlessRenderer`).
//!
//! Covers the Linux/Vulkan and macOS/Metal-via-wgpu paths used in production
//! export. Skips when no adapter is available.
//!
//! ```text
//! cargo test --features native-codec --test h264_gpu -- --nocapture
//! ```
#![cfg(feature = "native-codec")]

use threers::{
    encode_animation_rgba, AnimationEncodeOptions, BasicMaterial, BrowserCodec, Color,
    HeadlessRenderer, Material, Mesh, Object3D, OrthographicCamera, PlaneGeometry, Scene, Vector3,
};

const W: u32 = 64;
const H: u32 = 48;

fn renderer() -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder().size(W, H).build().ok()
}

fn unit_camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 2.0);
    cam.target = Vector3::ZERO;
    cam
}

fn encode_mp4(frames: impl IntoIterator<Item = Vec<u8>>) -> Vec<u8> {
    encode_animation_rgba(
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
    .expect("gpu rgba → mp4")
}

#[test]
fn wgpu_solid_frames_encode_mp4() {
    let Some(mut hr) = renderer() else {
        eprintln!("skipping h264_gpu: no wgpu adapter");
        return;
    };

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
        frames.push(hr.render_to_rgba(&mut scene, &cam));
        assert_eq!(frames[i].len(), (W * H * 4) as usize);
    }

    let mp4 = encode_mp4(frames);
    assert_eq!(&mp4[4..8], b"ftyp");
    assert!(mp4.len() > 200, "mp4 too small: {}", mp4.len());
    eprintln!("wgpu H.264 MP4 OK: {W}x{H} x4 ({} bytes)", mp4.len());
}
