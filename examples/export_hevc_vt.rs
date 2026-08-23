//! Headless cube → HEVC MP4 via Apple VideoToolbox (`hevc_videotoolbox`).
//!
//! ```text
//! cargo run --release --example export_hevc_vt --features video
//! ```

include!("video_common.inc");

fn main() {
    run_export(
        VideoCodec::HevcVideoToolbox,
        "./out/cube.hevc_vt.mp4",
        false,
        None,
    );
}
