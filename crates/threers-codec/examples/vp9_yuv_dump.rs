//! Decode a VP9 WebM or IVF file with `codec::decode::vp9` and write every
//! shown frame as raw planes in its native format (8-bit: one byte per
//! sample; 10/12-bit: little-endian u16), as `ffmpeg -f rawvideo` does.
//!
//!   vp9_yuv_dump IN.webm|IN.ivf OUT.yuv
use std::io::Write;

use threers_codec::decode::vp9::Vp9Decoder;
use threers_codec::mkv_demux::{parse_ivf, Mkv, TrackKind};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).expect("read");
    let packets: Vec<Vec<u8>> = if data.starts_with(b"DKIF") {
        parse_ivf(&data).expect("ivf").4.into_iter().map(|(_, d)| d).collect()
    } else {
        let mkv = Mkv::parse(&data).expect("mkv");
        let t = mkv.track(TrackKind::Video).expect("no video track");
        mkv.frames_of(t.number).map(|f| f.data.clone()).collect()
    };
    let mut dec = Vp9Decoder::new();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[2]).expect("create"));
    let mut n = 0;
    let mut desc = String::new();
    for p in &packets {
        let frames = match dec.decode(p) {
            Ok(f) => f,
            Err(e) => {
                println!("ERROR {e} (after {n} frames)");
                std::process::exit(2);
            }
        };
        for f in frames {
            for pl in 0..3 {
                let (s, _, _) = f.cropped(pl);
                if f.bpp == 8 {
                    out.write_all(&s.iter().map(|&v| v as u8).collect::<Vec<_>>()).unwrap();
                } else {
                    out.write_all(&s.iter().flat_map(|&v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
                }
            }
            desc = format!("{}x{} {}-bit ss {}{}", f.width, f.height, f.bpp, f.ss_h, f.ss_v);
            n += 1;
        }
    }
    println!("{n} frames, {desc}");
}
