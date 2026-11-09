//! Headless cube → animated GIF (native `GifWriter`, no ffmpeg).
//!
//! ```text
//! cargo run --release --example export_gif --features "video,native-codec"
//! ```

include!("video_common.inc");

fn main() {
    run_export(VideoCodec::Gif, "./out/cube.gif", true, Some(64));
}
