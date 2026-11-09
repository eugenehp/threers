//! Decode an MP4's video track and write raw planar YUV (8-bit as u8, else
//! u16 LE) in presentation order, for comparison with
//! `ffmpeg -i IN -f rawvideo -pix_fmt yuv420p -fps_mode passthrough OUT`.
//!
//!   cargo run --release -p threers-codec --example video_yuv_dump -- IN OUT.yuv
use threers_codec::decode::NativeDecoder;
use threers_codec::mp4_demux::demux_video;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: video_yuv_dump IN OUT.yuv");
        std::process::exit(2);
    }
    let data = std::fs::read(&a[1]).expect("read");
    let track = demux_video(&data).expect("demux");
    let mut dec = NativeDecoder::new(&track).expect("decoder");
    // Each sample's picture is decoded on push and queued under its pts.
    let mut frames = std::collections::BTreeMap::new();
    for s in &track.samples {
        let bytes = &data[s.offset as usize..s.offset as usize + s.size as usize];
        dec.push(bytes, s.pts, s.dts, s.duration as i64)
            .expect("decode");
        if let Some(f) = dec.take_frame(s.pts) {
            frames.insert(s.pts, f);
        }
    }
    let mut out = Vec::new();
    let mut n = 0;
    for f in frames.values() {
        for plane in [&f.y, &f.u, &f.v] {
            if f.bit_depth == 8 {
                out.extend(plane.iter().map(|&v| v as u8));
            } else {
                out.extend(plane.iter().flat_map(|&v| v.to_le_bytes()));
            }
        }
        n += 1;
    }
    eprintln!(
        "{n} frames of {} samples, {}x{}",
        track.samples.len(),
        track.width,
        track.height
    );
    std::fs::write(&a[2], out).expect("write");
}
