//! Headless cube → VP9 WebM with alpha (ffmpeg `yuva420p`).
//!
//! ```text
//! cargo run --release --example export_vp9_alpha --features video
//! ```

include!("video_common.inc");

fn main() {
    run_export(VideoCodec::Vp9, "./out/cube.vp9_alpha.webm", true, None);
}
