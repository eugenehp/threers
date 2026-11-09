//! Headless cube → H.264 MP4 (ffmpeg `libx264`).
//!
//! ```text
//! cargo run --release --example export_h264 --features video
//! cargo run --release --example export_h264 --features video -- --out /tmp/cube.mp4
//! ```

include!("video_common.inc");

fn main() {
    run_export(VideoCodec::H264, "./out/cube.h264.mp4", false, None);
}
