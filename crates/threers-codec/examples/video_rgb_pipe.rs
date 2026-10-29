//! Decode an MP4/MOV the way moviepy's ffmpeg pipe does
//! (`-vf scale=W:H -sws_flags bicubic -pix_fmt rgb24 -f image2pipe`) with
//! `codec::ffcompat::reader`, writing every piped frame as rgb24, or
//! comparing with such a dump.
//!
//!   video_rgb_pipe IN OUT.rgb
//!   video_rgb_pipe IN --compare REF.rgb
use std::collections::BTreeSet;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = threers_codec::ffcompat::reader::VideoReader::open(std::fs::read(&a[1]).expect("read")).unwrap_or_else(|e| {
        println!("ERROR {e}");
        std::process::exit(2)
    });
    let frames = r.pipe(None);
    let wanted: BTreeSet<usize> = frames.iter().copied().collect();
    let mut got = std::collections::BTreeMap::new();
    if let Err(e) = r.decode_rgb(&wanted, |i, rgb| {
        got.insert(i, rgb.to_vec());
        Ok(())
    }) {
        println!("ERROR {e}");
        std::process::exit(2)
    }
    let out: Vec<u8> = frames.iter().flat_map(|i| got[i].clone()).collect();
    let (w, h) = r.size();
    if a.get(2).map(String::as_str) == Some("--compare") {
        let reference = std::fs::read(&a[3]).expect("reference");
        let n = out.len().min(reference.len());
        let diff = (0..n).filter(|&i| out[i] != reference[i]).count();
        let maxd = (0..n).map(|i| (out[i] as i32 - reference[i] as i32).abs()).max().unwrap_or(0);
        let first = (0..n).find(|&i| out[i] != reference[i]).map(|i| (i / (w as usize * h as usize * 3), (i / 3) % w as usize, (i / 3 / w as usize) % h as usize, i % 3));
        println!("{w}x{h} frames ours {} ref {}: {diff} bytes differ (max {maxd}), first (frame, x, y, c) {first:?}", out.len() / (w as usize * h as usize * 3), reference.len() / (w as usize * h as usize * 3));
    } else {
        std::fs::write(&a[2], out).expect("write");
    }
}
