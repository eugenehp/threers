//! Headless cube → HEVC MP4 (ffmpeg `libx265`).
//!
//! ```text
//! cargo run --release --example export_hevc --features video
//! ```

include!("video_common.inc");

fn main() {
    run_export(VideoCodec::Hevc, "./out/cube.hevc.mp4", false, None);
}
