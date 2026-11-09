//! Headless cube → VP9 WebM (ffmpeg `libvpx-vp9`, opaque).
//!
//! ```text
//! cargo run --release --example export_vp9 --features video
//! ```

include!("video_common.inc");

fn main() {
    run_export(VideoCodec::Vp9, "./out/cube.vp9.webm", false, None);
}
