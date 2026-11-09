//! `ScadVideo`: an OpenSCAD animation straight to a video file.

#![cfg(not(target_arch = "wasm32"))]

use threers::openscad::animate::{ScadAnimation, ScadRender};
use threers_video::ScadVideo;

const GROWING: &str = "cube(10 + 40 * $t, center = true);";

fn gpu_available() -> bool {
    threers::HeadlessRenderer::builder()
        .size(16, 16)
        .build()
        .is_ok()
}

#[test]
fn video_export_picks_the_codec_from_the_extension() {
    if !gpu_available() {
        eprintln!("skipping: no GPU adapter");
        return;
    }
    let dir = std::env::temp_dir().join(format!("threers-scad-vid-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut a = ScadAnimation::from_source(GROWING).frames(4).fps(8);
    let out = dir.join("anim.gif");
    ScadRender::new(120, 80)
        .export_video(&mut a, &out)
        .expect("gif export");
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(&bytes[..3], b"GIF");
    let (_, frames) = threers::decode_gif(&bytes).expect("decode");
    assert_eq!(frames.len(), 4);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_encoded_video_uses_the_adjusted_frame_rate() {
    if !gpu_available() {
        eprintln!("skipping: no GPU adapter");
        return;
    }
    let dir = std::env::temp_dir().join(format!("threers-scad-speed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // 8 frames at 8 fps, played at 2× → 16 fps → a half-second loop.
    let mut a = ScadAnimation::from_source(GROWING)
        .frames(8)
        .fps(8)
        .speed(2.0);
    assert_eq!(a.frame_rate(), 16);
    let out = dir.join("fast.gif");
    ScadRender::new(96, 72).export_video(&mut a, &out).unwrap();

    let (_, frames) = threers::decode_gif(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(frames.len(), 8);
    // GIF delays are centiseconds: 16 fps ≈ 6 cs, not the 12 cs of 8 fps.
    assert!(frames[0].delay_num <= 7, "delay {} cs", frames[0].delay_num);
    let _ = std::fs::remove_dir_all(&dir);
}
