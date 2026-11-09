//! Headless cube → animated PNG (native `ApngEncoder`, no ffmpeg).
//!
//! ```text
//! cargo run --release --example export_apng --features "video,native-codec"
//! ```

include!("video_common.inc");

fn main() {
    run_export(VideoCodec::Apng, "./out/cube.apng.png", true, None);
}
