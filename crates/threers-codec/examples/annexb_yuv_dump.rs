//! Decode an Annex B elementary stream (`.264` / `.265`, start-code
//! delimited) and write its pictures as raw planar YUV in display order
//! (8-bit as u8, else u16 LE), for comparison with
//! `ffmpeg -i IN -f rawvideo -pix_fmt FMT -fps_mode passthrough OUT`.
//!
//!   annexb_yuv_dump h264|hevc IN OUT.yuv
use threers_codec::decode::decode_annexb;
use threers_codec::mp4_demux::TrackCodec;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let codec = if a[1] == "hevc" { TrackCodec::Hevc } else { TrackCodec::H264 };
    let data = std::fs::read(&a[2]).expect("read");
    let frames = decode_annexb(codec, &data).unwrap_or_else(|e| {
        println!("ERROR {e}");
        std::process::exit(2)
    });
    let mut out = Vec::new();
    for f in &frames {
        for plane in [&f.y, &f.u, &f.v] {
            if f.bit_depth == 8 {
                out.extend(plane.iter().map(|&v| v as u8));
            } else {
                out.extend(plane.iter().flat_map(|&v| v.to_le_bytes()));
            }
        }
    }
    std::fs::write(&a[3], out).expect("write");
    if let Some(f) = frames.first() {
        println!("{} frames, {}x{} {:?} {}-bit", frames.len(), f.width, f.height, f.chroma, f.bit_depth);
    }
}
